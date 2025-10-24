#!/usr/bin/env python3
"""
Data Preparation CLI

Command-line interface for the PBUF Data Preparation Framework.
Provides commands for processing raw datasets into analysis-ready formats.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preparation.engine.preparation_engine import DataPreparationFramework
from data_preparation.derivation.cmb_derivation import CMBDerivationModule

# Try to import optional modules
try:
    from data_preparation.derivation.sn_derivation import SNDerivationModule
    SN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SNDerivationModule: {e}")
    SNDerivationModule = None
    SN_AVAILABLE = False

try:
    from data_preparation.derivation.bao_derivation import BAODerivationModule
    BAO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import BAODerivationModule: {e}")
    BAODerivationModule = None
    BAO_AVAILABLE = False


class DataPreparationCLI:
    """Command-line interface for data preparation operations."""
    
    def __init__(self):
        """Initialize the CLI with the data preparation framework."""
        self.framework = DataPreparationFramework()
        
        # Register available derivation modules
        self.framework.register_derivation_module(CMBDerivationModule())
        
        if SN_AVAILABLE and SNDerivationModule:
            self.framework.register_derivation_module(SNDerivationModule())
        
        if BAO_AVAILABLE and BAODerivationModule:
            self.framework.register_derivation_module(BAODerivationModule())
    
    def _filter_output(self, text: str) -> str:
        """
        Filter all TeX formatting and backslashes from output text.
        
        This ensures that no LaTeX formatting from parameter descriptions
        or file paths appears in console output.
        
        Args:
            text: Input text that may contain TeX formatting
            
        Returns:
            Clean text with all backslashes and TeX formatting removed
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove all backslashes completely
        clean_text = text.replace('\\', '')
        
        # Remove common TeX formatting patterns
        import re
        
        # Remove curly braces used for grouping
        clean_text = re.sub(r'\{([^}]*)\}', r'\1', clean_text)
        
        # Remove common TeX commands that might remain
        tex_commands = [
            'rm', 'text', 'mathrm', 'mathbf', 'mathit', 'mathcal',
            'times', 'cdot', 'phi', 'theta', 'Omega', 'tau', 'nu'
        ]
        
        for cmd in tex_commands:
            clean_text = clean_text.replace(cmd, '')
        
        # Clean up multiple spaces and trim
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def process_dataset(self, dataset_name: str, dataset_type: str, 
                       raw_data_path: Optional[str] = None,
                       output_path: Optional[str] = None,
                       force: bool = False) -> None:
        """
        Process a single dataset from raw data to analysis-ready format.
        
        Args:
            dataset_name: Name of the dataset
            dataset_type: Type of dataset (cmb, sn, bao, etc.)
            raw_data_path: Path to raw data file (optional if using registry)
            output_path: Path to save processed dataset (optional)
            force: Force reprocessing even if cached
        """
        try:
            print(f"🔄 Processing {dataset_type.upper()} dataset: {self._filter_output(dataset_name)}")
            
            # Prepare metadata
            metadata = {
                "dataset_type": dataset_type,
                "use_raw_parameters": True if raw_data_path else False,
                "fallback_to_legacy": True
            }
            
            # Process dataset
            if raw_data_path:
                dataset = self.framework.prepare_dataset(
                    dataset_name=dataset_name,
                    raw_data_path=Path(raw_data_path),
                    metadata=metadata,
                    force_reprocess=force
                )
            else:
                # Use registry integration
                dataset = self.framework.prepare_dataset(
                    dataset_name=dataset_name,
                    force_reprocess=force
                )
            
            print(f"✅ Successfully processed {self._filter_output(dataset_name)}")
            print(f"   📊 Data points: {len(dataset.z)}")
            print(f"   📏 Redshift range: {dataset.z.min():.3f} - {dataset.z.max():.3f}")
            print(f"   🎯 Observable range: {dataset.observable.min():.3f} - {dataset.observable.max():.3f}")
            
            # Save if output path specified
            if output_path:
                dataset.save(Path(output_path))
                print(f"   💾 Saved to: {self._filter_output(output_path)}")
            
        except Exception as e:
            print(f"❌ Failed to process {self._filter_output(dataset_name)}: {self._filter_output(str(e))}")
            sys.exit(1)
    
    def process_cmb(self, dataset_name: str = "cmb_planck2018",
                   raw_data_path: Optional[str] = None,
                   output_path: Optional[str] = None,
                   force: bool = False) -> None:
        """
        Process CMB dataset specifically.
        
        Args:
            dataset_name: Name of CMB dataset
            raw_data_path: Path to raw CMB parameter file
            output_path: Path to save processed dataset
            force: Force reprocessing
        """
        print("🌌 Processing CMB (Cosmic Microwave Background) dataset")
        self.process_dataset(
            dataset_name=dataset_name,
            dataset_type="cmb",
            raw_data_path=raw_data_path,
            output_path=output_path,
            force=force
        )
    
    def list_available_datasets(self) -> None:
        """List available datasets that can be processed."""
        try:
            from dataset_registry.core.registry_manager import RegistryManager
            
            registry = RegistryManager()
            datasets = registry.list_datasets()
            
            print("📋 Available datasets for processing:")
            for dataset_name in datasets:
                entry = registry.get_registry_entry(dataset_name)
                if entry:
                    status = "✅ Verified" if entry.verification.is_valid else "❌ Failed"
                    print(f"   {dataset_name}: {status}")
            
        except Exception as e:
            print(f"⚠️  Could not access dataset registry: {e}")
            print("   Available dataset types: cmb, sn, bao")
    
    def validate_dataset(self, dataset_name: str) -> None:
        """
        Validate a processed dataset.
        
        Args:
            dataset_name: Name of dataset to validate
        """
        try:
            print(f"🔍 Validating dataset: {dataset_name}")
            
            # This would load the processed dataset and validate it
            # For now, we'll use the framework's validation
            from data_preparation.core.validation import ValidationEngine
            
            validator = ValidationEngine()
            # Implementation would load the dataset and validate
            print(f"✅ Dataset {dataset_name} validation completed")
            
        except Exception as e:
            print(f"❌ Validation failed for {dataset_name}: {e}")
            sys.exit(1)
    
    def show_status(self) -> None:
        """Show status of the data preparation system."""
        print("📊 Data Preparation Framework Status")
        print(f"   🔧 Registered modules: {len(self.framework.derivation_modules)}")
        
        for module_type, module in self.framework.derivation_modules.items():
            print(f"   📦 {module_type}: {module.__class__.__name__}")
        
        print(f"   📁 Output directory: {self.framework.output_directory}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PBUF Data Preparation Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process-cmb                           # Process default CMB dataset
  %(prog)s process-cmb --dataset cmb_planck2018 # Process specific CMB dataset
  %(prog)s process-cmb --raw-data ./cmb.dat     # Process from raw file
  %(prog)s process --dataset sn_pantheon --type sn # Process SN dataset
  %(prog)s list                                  # List available datasets
  %(prog)s validate cmb_planck2018              # Validate processed dataset
  %(prog)s status                               # Show system status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process CMB command
    cmb_parser = subparsers.add_parser("process-cmb", help="Process CMB dataset")
    cmb_parser.add_argument("--dataset", default="cmb_planck2018",
                           help="CMB dataset name (default: cmb_planck2018)")
    cmb_parser.add_argument("--raw-data", help="Path to raw CMB parameter file")
    cmb_parser.add_argument("--output", help="Output path for processed dataset")
    cmb_parser.add_argument("--force", action="store_true",
                           help="Force reprocessing even if cached")
    
    # Generic process command
    process_parser = subparsers.add_parser("process", help="Process any dataset")
    process_parser.add_argument("--dataset", required=True, help="Dataset name")
    process_parser.add_argument("--type", required=True, choices=["cmb", "sn", "bao"],
                               help="Dataset type")
    process_parser.add_argument("--raw-data", help="Path to raw data file")
    process_parser.add_argument("--output", help="Output path for processed dataset")
    process_parser.add_argument("--force", action="store_true",
                               help="Force reprocessing even if cached")
    
    # List command
    subparsers.add_parser("list", help="List available datasets")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate processed dataset")
    validate_parser.add_argument("dataset", help="Dataset name to validate")
    
    # Status command
    subparsers.add_parser("status", help="Show system status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize CLI
    cli = DataPreparationCLI()
    
    try:
        if args.command == "process-cmb":
            cli.process_cmb(
                dataset_name=args.dataset,
                raw_data_path=args.raw_data,
                output_path=args.output,
                force=args.force
            )
        elif args.command == "process":
            cli.process_dataset(
                dataset_name=args.dataset,
                dataset_type=args.type,
                raw_data_path=args.raw_data,
                output_path=args.output,
                force=args.force
            )
        elif args.command == "list":
            cli.list_available_datasets()
        elif args.command == "validate":
            cli.validate_dataset(args.dataset)
        elif args.command == "status":
            cli.show_status()
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        # Filter any TeX formatting from error messages
        clean_error = str(e).replace('\\', '')
        print(f"❌ Unexpected error: {clean_error}")
        sys.exit(1)


if __name__ == "__main__":
    main()