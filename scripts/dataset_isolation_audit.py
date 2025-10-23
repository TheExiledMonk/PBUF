#!/usr/bin/env python3
"""
Dataset Isolation Audit Script

Simple audit to verify that each fit_* module only accesses its designated datasets
and doesn't load any unrelated datasets during execution.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class AuditResult:
    module_name: str
    allowed_datasets: Set[str]
    accessed_datasets: Set[str]
    violations: List[str]
    warnings: List[str]
    status: str  # "PASS" or "VIOLATION"


# Dataset whitelist for each fit module (using actual dataset names from code)
DATASET_WHITELIST = {
    "fit_sn": {"sn"},  # maps to sn_pantheon_plus
    "fit_bao": {"bao"},  # maps to bao_dr16_isotropic  
    "fit_bao_aniso": {"bao_ani"},  # maps to bao_dr16_anisotropic
    "fit_cmb": {"cmb"},  # maps to cmb_planck2018_distance_priors
    "fit_joint": {"cmb", "sn", "bao", "bao_ani"}  # joint fit allows multiple datasets
}

# Known dataset identifiers to look for in code
KNOWN_DATASETS = {
    "sn_pantheon_plus",
    "bao_dr16_isotropic", 
    "bao_dr16_anisotropic",
    "cmb_planck2018_distance_priors",
    "cc_moresco2022",
    "rsd_nesseris2017",
    # Legacy/alternative names
    "cmb", "bao", "bao_ani", "sn", "cc", "rsd",
    "cmb_planck2018", "bao_compilation", "bao_aniso_boss_dr12"
}


@dataclass
class DatasetAccess:
    dataset_name: str
    access_type: str  # "REQUESTED", "RECEIVED", "REFERENCED"
    location: str     # Where in code this was found
    line_number: int = 0


class DatasetAccessAnalyzer(ast.NodeVisitor):
    """AST visitor to find dataset access patterns in Python code."""
    
    def __init__(self):
        self.dataset_accesses = []
        self.function_calls = []
        self.string_literals = []
        self.current_line = 0
        
    def visit_Call(self, node):
        """Visit function calls to detect dataset loading."""
        self.current_line = getattr(node, 'lineno', 0)
        
        # Check for direct load_dataset() calls - REQUESTED
        if isinstance(node.func, ast.Name) and node.func.id == "load_dataset":
            if node.args and isinstance(node.args[0], ast.Constant):
                self.dataset_accesses.append(DatasetAccess(
                    dataset_name=node.args[0].value,
                    access_type="REQUESTED",
                    location=f"load_dataset() call",
                    line_number=self.current_line
                ))
        
        # Check for engine.run_fit() calls with datasets_list - REQUESTED
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == "run_fit"):
            for keyword in node.keywords:
                if keyword.arg == "datasets_list":
                    if isinstance(keyword.value, ast.List):
                        for elt in keyword.value.elts:
                            if isinstance(elt, ast.Constant):
                                self.dataset_accesses.append(DatasetAccess(
                                    dataset_name=elt.value,
                                    access_type="REQUESTED",
                                    location=f"engine.run_fit(datasets_list=...)",
                                    line_number=self.current_line
                                ))
        
        # Check for integrity checks that might auto-add datasets - RECEIVED
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == "run_integrity_suite"):
            for keyword in node.keywords:
                if keyword.arg == "datasets":
                    if isinstance(keyword.value, ast.List):
                        for elt in keyword.value.elts:
                            if isinstance(elt, ast.Constant):
                                self.dataset_accesses.append(DatasetAccess(
                                    dataset_name=elt.value,
                                    access_type="REQUESTED",
                                    location=f"integrity.run_integrity_suite(datasets=...)",
                                    line_number=self.current_line
                                ))
        
        # Check for compute_dof() calls that might auto-add datasets - RECEIVED
        if (isinstance(node.func, ast.Name) and node.func.id == "compute_dof"):
            if node.args and isinstance(node.args[0], ast.List):
                for elt in node.args[0].elts:
                    if isinstance(elt, ast.Constant):
                        self.dataset_accesses.append(DatasetAccess(
                            dataset_name=elt.value,
                            access_type="REQUESTED",
                            location=f"compute_dof() call",
                            line_number=self.current_line
                        ))
        
        # Record all function calls for manual review
        if isinstance(node.func, ast.Name):
            self.function_calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.function_calls.append(node.func.attr)
            
        self.generic_visit(node)
    
    def visit_Constant(self, node):
        """Visit string constants that might be dataset names."""
        if isinstance(node.value, str):
            self.string_literals.append(node.value)
            # Check if this constant is a known dataset name
            if node.value in KNOWN_DATASETS:
                self.dataset_accesses.append(DatasetAccess(
                    dataset_name=node.value,
                    access_type="REFERENCED",
                    location="string literal",
                    line_number=getattr(node, 'lineno', 0)
                ))
        self.generic_visit(node)


def analyze_file_for_datasets(file_path: Path) -> Tuple[List[DatasetAccess], List[str]]:
    """
    Analyze a Python file for dataset access patterns.
    
    Returns:
        Tuple of (dataset_accesses, potential_issues)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        return [], [f"Failed to read file: {e}"]
    
    # Parse AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return [], [f"Syntax error in file: {e}"]
    
    # Analyze with AST visitor
    analyzer = DatasetAccessAnalyzer()
    analyzer.visit(tree)
    
    dataset_accesses = analyzer.dataset_accesses.copy()
    potential_issues = []
    
    # Check for hardcoded dataset references in comments and strings
    dataset_patterns = [
        r'pantheon[_\+]?plus',
        r'dr16[_\-]?isotropic',
        r'dr16[_\-]?anisotropic', 
        r'planck2018[_\-]?distance[_\-]?priors',
        r'moresco2022',
        r'nesseris2017'
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern in dataset_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            if matches:
                # Try to map pattern back to canonical dataset name
                pattern_lower = pattern.lower()
                canonical_name = None
                if 'pantheon' in pattern_lower:
                    canonical_name = 'sn_pantheon_plus'
                elif 'dr16' in pattern_lower and 'isotropic' in pattern_lower:
                    canonical_name = 'bao_dr16_isotropic'
                elif 'dr16' in pattern_lower and 'anisotropic' in pattern_lower:
                    canonical_name = 'bao_dr16_anisotropic'
                elif 'planck2018' in pattern_lower:
                    canonical_name = 'cmb_planck2018_distance_priors'
                elif 'moresco' in pattern_lower:
                    canonical_name = 'cc_moresco2022'
                elif 'nesseris' in pattern_lower:
                    canonical_name = 'rsd_nesseris2017'
                
                if canonical_name:
                    dataset_accesses.append(DatasetAccess(
                        dataset_name=canonical_name,
                        access_type="REFERENCED",
                        location=f"pattern match in line: {line.strip()}",
                        line_number=i
                    ))
    
    # Look for automatic dataset addition patterns
    for i, line in enumerate(lines, 1):
        # Check for DOF-based automatic dataset addition
        if 'add' in line.lower() and any(ds in line.lower() for ds in ['bao', 'cmb']):
            if 'dof' in line.lower() or 'degrees' in line.lower():
                potential_issues.append(f"Line {i}: Possible automatic dataset addition for DOF: {line.strip()}")
        
        # Check for integrity check auto-addition
        if 'integrity' in line.lower() and any(ds in line.lower() for ds in ['bao', 'cmb', 'sn']):
            potential_issues.append(f"Line {i}: Integrity check may auto-add datasets: {line.strip()}")
    
    return dataset_accesses, potential_issues


@dataclass
class DetailedAuditResult:
    module_name: str
    allowed_datasets: Set[str]
    dataset_accesses: List[DatasetAccess]
    violations: List[str]
    warnings: List[str]
    status: str  # "PASS" or "VIOLATION"
    
    @property
    def accessed_datasets(self) -> Set[str]:
        return {access.dataset_name for access in self.dataset_accesses}


def audit_fit_module(module_path: Path) -> DetailedAuditResult:
    """Audit a single fit module for dataset isolation compliance."""
    
    module_name = module_path.stem
    allowed_datasets = DATASET_WHITELIST.get(module_name, set())
    
    dataset_accesses, issues = analyze_file_for_datasets(module_path)
    
    violations = []
    warnings = []
    
    # Analyze each dataset access
    accessed_dataset_names = {access.dataset_name for access in dataset_accesses}
    
    # Check for unauthorized dataset access
    unauthorized = accessed_dataset_names - allowed_datasets
    for dataset in unauthorized:
        # Find how this dataset was accessed
        accesses = [a for a in dataset_accesses if a.dataset_name == dataset]
        for access in accesses:
            if access.access_type == "REQUESTED":
                violations.append(f"violation: module actively requests unauthorized dataset '{dataset}' via {access.location} (line {access.line_number})")
            elif access.access_type == "RECEIVED":
                violations.append(f"violation: module receives unauthorized dataset '{dataset}' via {access.location} (line {access.line_number})")
            else:
                warnings.append(f"warning: unauthorized dataset '{dataset}' referenced in {access.location} (line {access.line_number})")
    
    # Check for potential cross-references (datasets mentioned but not in whitelist)
    for access in dataset_accesses:
        if access.dataset_name not in KNOWN_DATASETS:
            warnings.append(f"warning: potential cross-reference to unknown dataset '{access.dataset_name}' in {access.location} (line {access.line_number})")
    
    # Add any file analysis issues
    warnings.extend(issues)
    
    status = "VIOLATION" if violations else "PASS"
    
    return DetailedAuditResult(
        module_name=module_name,
        allowed_datasets=allowed_datasets,
        dataset_accesses=dataset_accesses,
        violations=violations,
        warnings=warnings,
        status=status
    )


def run_full_audit() -> Dict[str, DetailedAuditResult]:
    """Run complete audit of all fit modules."""
    
    results = {}
    
    # Find all fit_* modules
    fit_modules = [
        Path("pipelines/fit_sn.py"),
        Path("pipelines/fit_bao.py"), 
        Path("pipelines/fit_cmb.py"),
        Path("pipelines/fit_joint.py"),
        Path("pipelines/fit_bao_aniso.py")
    ]
    
    for module_path in fit_modules:
        if module_path.exists():
            result = audit_fit_module(module_path)
            results[result.module_name] = result
        else:
            print(f"⚠️  Module not found: {module_path}")
    
    return results


def generate_audit_report(results: Dict[str, DetailedAuditResult]) -> str:
    """Generate comprehensive audit report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATASET ISOLATION AUDIT REPORT")
    report_lines.append("=" * 80)
    
    # Overall summary
    total_modules = len(results)
    passed_modules = sum(1 for r in results.values() if r.status == "PASS")
    violation_modules = total_modules - passed_modules
    
    if violation_modules == 0:
        report_lines.append("✅ All fit modules confirmed isolated")
    else:
        report_lines.append(f"❌ {violation_modules} module(s) have dataset isolation violations")
    
    report_lines.append(f"\nSummary: {passed_modules}/{total_modules} modules passed")
    report_lines.append("")
    
    # Per-module results
    for module_name, result in sorted(results.items()):
        status_icon = "✅" if result.status == "PASS" else "❌"
        report_lines.append(f"{status_icon} {module_name}")
        
        # Show allowed datasets
        if result.allowed_datasets:
            allowed_list = ", ".join(sorted(result.allowed_datasets))
            report_lines.append(f"   Allowed: {allowed_list}")
        else:
            report_lines.append(f"   Allowed: (no whitelist defined)")
        
        # Show accessed datasets with access type breakdown
        if result.dataset_accesses:
            requested = [a for a in result.dataset_accesses if a.access_type == "REQUESTED"]
            referenced = [a for a in result.dataset_accesses if a.access_type == "REFERENCED"]
            
            if requested:
                req_datasets = {a.dataset_name for a in requested}
                report_lines.append(f"   Actively Requests: {', '.join(sorted(req_datasets))}")
            
            if referenced:
                ref_datasets = {a.dataset_name for a in referenced}
                report_lines.append(f"   References: {', '.join(sorted(ref_datasets))}")
        else:
            report_lines.append(f"   Accessed: (none detected)")
        
        # Show violations with details
        if result.violations:
            report_lines.append(f"   Violations:")
            for violation in result.violations:
                report_lines.append(f"     • {violation}")
        
        # Show warnings
        if result.warnings:
            report_lines.append(f"   Warnings:")
            for warning in result.warnings:
                report_lines.append(f"     • {warning}")
        
        report_lines.append("")
    
    # Dataset access summary
    report_lines.append("Dataset Access Summary:")
    report_lines.append("-" * 40)
    
    all_datasets_accessed = set()
    for result in results.values():
        all_datasets_accessed.update(result.accessed_datasets)
    
    for dataset in sorted(all_datasets_accessed):
        accessing_modules = [
            r.module_name for r in results.values() 
            if dataset in r.accessed_datasets
        ]
        modules_str = ", ".join(accessing_modules)
        report_lines.append(f"  {dataset}: {modules_str}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def main():
    """Main audit execution."""
    print("Running Dataset Isolation Audit...")
    print()
    
    # Run the audit
    results = run_full_audit()
    
    # Generate and display report
    report = generate_audit_report(results)
    print(report)
    
    # Save report to file
    with open("dataset_isolation_audit_report.txt", "w") as f:
        f.write(report)
    
    print(f"Report saved to: dataset_isolation_audit_report.txt")
    
    # Return exit code based on results
    violations_found = any(r.status == "VIOLATION" for r in results.values())
    return 1 if violations_found else 0


if __name__ == "__main__":
    exit(main())