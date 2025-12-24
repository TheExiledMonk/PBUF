#!/bin/bash
source .venv/bin/activate
python cosmos_cli.py science --config config/science_runs/unified_joint-1.json --monitor
python cosmos_cli.py science --config config/science_runs/unified_joint-2.json --monitor
python cosmos_cli.py science --config config/science_runs/unified_joint-3.json --monitor
python cosmos_cli.py science --config config/science_runs/unified_joint-4.json --monitor
python cosmos_cli.py science --config config/science_runs/unified_joint-5.json --monitor
