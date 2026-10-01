"""
GenRec CLI - Command Line Interface
Built by: Sandesh Dhital
"""

import argparse
import json
import sys
from .verbalizer import Verbalizer
from .context_engine import ContextEngine
from .ranker import GenRecRanker
from .reward import RewardModel
from .serving import GenRecServer, ServingConfig
from .evaluation import Evaluator


def main():
    parser = argparse.ArgumentParser(description='GenRec: LLM-Backed Recommendation Ranker')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a quick demo')
    demo_parser.add_argument('--openai', action='store_true', help='Use OpenAI API')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start the web server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    serve_parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    serve_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate rankings')
    eval_parser.add_argument('--rankings', required=True, help='Path to rankings JSON file')
    eval_parser.add_argument('--grades', required=True, help='Path to relevance grades JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        run_demo(args)
    elif args.command == 'serve':
        run_server(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    else:
        parser.print_help()


def run_demo(args):
    """Run demo"""
    from demo import main as demo_main
    demo_main()


def run_server(args):
    """Run web server"""
    from app import app
    app.run(host=args.host, port=args.port, debug=args.debug)


def run_evaluation(args):
    """Run evaluation"""
    with open(args.rankings) as f:
        rankings = json.load(f)
    with open(args.grades) as f:
        grades = json.load(f)
    
    evaluator = Evaluator()
    metrics = evaluator.evaluate_offline(rankings, grades)
    report = evaluator.generate_report(metrics)
    print(report)


if __name__ == '__main__':
    main()
