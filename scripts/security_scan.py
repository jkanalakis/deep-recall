#!/usr/bin/env python3
"""
Security Scanner Script

This script performs comprehensive security scanning of the codebase,
including dependency vulnerabilities, code security issues, and hardcoded secrets.
It uses various security tools and generates detailed reports.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SecurityScanner:
    def __init__(
        self, project_root: str, config_path: str = "config/security_config.json"
    ):
        """
        Initialize the security scanner.

        Args:
            project_root: Root directory of the project to scan
            config_path: Path to the configuration file
        """
        self.project_root = Path(project_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.config = self._load_config()
        self.report_dir = Path(self.config["reporting"]["output_dir"])
        self.report_dir.mkdir(exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the configuration file."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {self.config_path}")
            sys.exit(1)

    def _install_required_tools(self) -> None:
        """Install required security scanning tools if not present."""
        tools = {
            "safety": "pip install safety",
            "bandit": "pip install bandit",
            "detect-secrets": "pip install detect-secrets",
        }

        for tool, install_cmd in tools.items():
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.info(f"Installing {tool}...")
                subprocess.run(install_cmd.split(), check=True)

    def scan_dependencies(self) -> Dict[str, Any]:
        """
        Scan Python dependencies for known vulnerabilities using safety.

        Returns:
            Dictionary containing scan results
        """
        if not self.config["scan_settings"]["dependencies"]["enabled"]:
            return {"status": "disabled", "vulnerabilities": []}

        logger.info("Scanning dependencies for vulnerabilities...")
        self._install_required_tools()

        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            vulnerabilities = json.loads(result.stdout)
            return {"status": "success", "vulnerabilities": vulnerabilities}
        except subprocess.CalledProcessError as e:
            logger.error(f"Dependency scan failed: {e}")
            return {"status": "error", "error": str(e), "vulnerabilities": []}

    def scan_code(self) -> Dict[str, Any]:
        """
        Scan Python code for security issues using bandit.

        Returns:
            Dictionary containing scan results
        """
        if not self.config["scan_settings"]["code"]["enabled"]:
            return {"status": "disabled", "issues": []}

        logger.info("Scanning code for security issues...")
        self._install_required_tools()

        try:
            result = subprocess.run(
                ["bandit", "-r", "-f", "json", str(self.project_root)],
                capture_output=True,
                text=True,
                check=True,
            )
            issues = json.loads(result.stdout)
            return {"status": "success", "issues": issues}
        except subprocess.CalledProcessError as e:
            logger.error(f"Code scan failed: {e}")
            return {"status": "error", "error": str(e), "issues": []}

    def scan_secrets(self) -> Dict[str, Any]:
        """
        Scan for hardcoded secrets using detect-secrets.

        Returns:
            Dictionary containing scan results
        """
        if not self.config["scan_settings"]["secrets"]["enabled"]:
            return {"status": "disabled", "secrets": []}

        logger.info("Scanning for hardcoded secrets...")
        self._install_required_tools()

        try:
            result = subprocess.run(
                ["detect-secrets", "scan", "--json", str(self.project_root)],
                capture_output=True,
                text=True,
                check=True,
            )
            secrets = json.loads(result.stdout)
            return {"status": "success", "secrets": secrets}
        except subprocess.CalledProcessError as e:
            logger.error(f"Secrets scan failed: {e}")
            return {"status": "error", "error": str(e), "secrets": []}

    def generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the scan results.

        Args:
            results: Dictionary containing all scan results

        Returns:
            Markdown formatted summary
        """
        summary = ["# Security Scan Summary\n"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary.append(f"Scan completed at: {timestamp}\n")

        # Dependencies section
        summary.append("## Dependencies\n")
        if results["dependencies"]["status"] == "success":
            vulns = results["dependencies"]["vulnerabilities"]
            if vulns:
                summary.append(f"Found {len(vulns)} vulnerabilities:\n")
                for vuln in vulns:
                    summary.append(f"- {vuln['package']}: {vuln['vulnerability']}")
            else:
                summary.append("No vulnerabilities found.\n")
        else:
            summary.append("Dependency scan was disabled or failed.\n")

        # Code section
        summary.append("\n## Code Security\n")
        if results["code"]["status"] == "success":
            issues = results["code"]["issues"]
            if issues:
                summary.append(f"Found {len(issues)} security issues:\n")
                for issue in issues:
                    summary.append(
                        f"- {issue['filename']}: {issue['issue_text']} "
                        f"(Severity: {issue['issue_severity']})"
                    )
            else:
                summary.append("No security issues found.\n")
        else:
            summary.append("Code scan was disabled or failed.\n")

        # Secrets section
        summary.append("\n## Hardcoded Secrets\n")
        if results["secrets"]["status"] == "success":
            secrets = results["secrets"]["secrets"]
            if secrets:
                summary.append(f"Found {len(secrets)} potential secrets:\n")
                for secret in secrets:
                    summary.append(f"- {secret['filename']}: {secret['type']}")
            else:
                summary.append("No hardcoded secrets found.\n")
        else:
            summary.append("Secrets scan was disabled or failed.\n")

        return "\n".join(summary)

    def save_reports(self, results: Dict[str, Any], summary: str) -> None:
        """
        Save scan results and summary to files.

        Args:
            results: Dictionary containing all scan results
            summary: Markdown formatted summary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        if "json" in self.config["reporting"]["formats"]:
            results_file = self.report_dir / f"security_scan_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")

        # Save Markdown summary
        if "markdown" in self.config["reporting"]["formats"]:
            summary_file = self.report_dir / f"security_scan_{timestamp}.md"
            with open(summary_file, "w") as f:
                f.write(summary)
            logger.info(f"Summary saved to {summary_file}")

    def run_scan(self, scan_type: str = "all") -> None:
        """
        Run the specified type of security scan.

        Args:
            scan_type: Type of scan to run ("dependencies", "code", "secrets", or "all")
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "scan_type": scan_type,
        }

        if scan_type in ["dependencies", "all"]:
            results["dependencies"] = self.scan_dependencies()

        if scan_type in ["code", "all"]:
            results["code"] = self.scan_code()

        if scan_type in ["secrets", "all"]:
            results["secrets"] = self.scan_secrets()

        summary = self.generate_summary(results)
        self.save_reports(results, summary)


def main():
    parser = argparse.ArgumentParser(description="Security Scanner")
    parser.add_argument(
        "--project-root", default=".", help="Root directory of the project to scan"
    )
    parser.add_argument(
        "--config",
        default="config/security_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--scan-type",
        choices=["dependencies", "code", "secrets", "all"],
        default="all",
        help="Type of scan to perform",
    )

    args = parser.parse_args()

    scanner = SecurityScanner(args.project_root, args.config)
    scanner.run_scan(args.scan_type)


if __name__ == "__main__":
    main()
