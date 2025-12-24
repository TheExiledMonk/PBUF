#!/usr/bin/env python3
"""
Standalone CLI for the reporting system - completely independent of cosmos2
"""

import argparse
import sys
from pathlib import Path
import logging

# Add the reporting system to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reporting_system.core.report_generator import ReportGenerator
from reporting_system.core.aggregated_report_generator import JackknifeAggregatedReportGenerator

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate beautiful reports from cosmos2 science runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python report_cli.py /path/to/science/run
  python report_cli.py /path/to/science/run --output my_report.html
  python report_cli.py /path/to/science/run --verbose
        """
    )
    
    parser.add_argument(
        "run_directory",
        help="Path to the science run directory to analyze"
    )

    parser.add_argument(
        "--aggregate-jackknife",
        action="store_true",
        help="Enable jackknife aggregation mode: treat multiple runs under run_directory as replicas and pool fold-level jackknife outputs.",
    )

    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        default=[],
        help="Explicit run directory to include when using --aggregate-jackknife (can be repeated).",
    )

    parser.add_argument(
        "--no-latest-per-run-name",
        action="store_true",
        help="When aggregating, do not auto-select only the latest directory per run_name.",
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path for the report (default: auto-generated in run directory)"
    )

    parser.add_argument(
        "--template", "-t",
        help="Path to the HTML template used for rendering the report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate run directory
    run_dir = Path(args.run_directory)
    if not run_dir.exists():
        print(f"❌ Error: Run directory does not exist: {run_dir}")
        sys.exit(1)
    
    if not run_dir.is_dir():
        print(f"❌ Error: Path is not a directory: {run_dir}")
        sys.exit(1)
    
    try:
        output_path = Path(args.output) if args.output else None

        if args.aggregate_jackknife:
            print(f"🚀 Generating JACKKNIFE-AGGREGATED report for: {run_dir.name}")
            generator = JackknifeAggregatedReportGenerator(
                run_dir,
                run_dirs=[Path(p) for p in args.runs] if args.runs else None,
                select_latest=not args.no_latest_per_run_name,
                template_path=Path(args.template) if args.template else None,
            )
            report_path = generator.generate_report(output_path)
        else:
            # Generate report
            print(f"🚀 Generating report for: {run_dir.name}")
            generator = ReportGenerator(run_dir, template_path=Path(args.template) if args.template else None)
            report_path = generator.generate_report(output_path)
        
        print(f"✅ Report generated successfully!")
        print(f"📁 Report location: {report_path}")
        
        # Show file size
        report_file = Path(report_path)
        file_size = report_file.stat().st_size
        print(f"📊 File size: {file_size:,} bytes")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
