#!/usr/bin/env python3
"""
Backup and disaster recovery script.

This script provides functionality to backup and restore data from
PostgreSQL and vector databases.
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)

logger = logging.getLogger(__name__)


class BackupManager:
    """
    A class to manage backups and recovery.
    """

    def __init__(self, config_path: str):
        """
        Initialize the backup manager.

        Args:
            config_path: Path to the backup configuration file
        """
        self.config = self._load_config(config_path)
        self.backup_dir = self.config.get("backup_dir", "backups")
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create backup directory
        os.makedirs(self.backup_dir, exist_ok=True)

        logger.info(f"Initialized backup manager with config: {config_path}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load backup configuration from file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary with configuration
        """
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

    def backup_postgresql(self) -> str:
        """
        Backup PostgreSQL database.

        Returns:
            Path to the backup file
        """
        logger.info("Backing up PostgreSQL database...")

        # Get PostgreSQL configuration
        pg_config = self.config.get("postgresql", {})
        db_name = pg_config.get("db_name", "deep_recall")
        db_user = pg_config.get("db_user", "postgres")
        db_host = pg_config.get("db_host", "localhost")
        db_port = pg_config.get("db_port", "5432")

        # Create backup filename
        backup_file = os.path.join(self.backup_dir, f"postgresql_{self.timestamp}.sql")

        # Run pg_dump
        try:
            cmd = [
                "pg_dump",
                "-h",
                db_host,
                "-p",
                db_port,
                "-U",
                db_user,
                "-d",
                db_name,
                "-f",
                backup_file,
            ]

            # Set PGPASSWORD environment variable if provided
            env = os.environ.copy()
            if "db_password" in pg_config:
                env["PGPASSWORD"] = pg_config["db_password"]

            subprocess.run(cmd, env=env, check=True)
            logger.info(f"PostgreSQL backup completed: {backup_file}")

            return backup_file

        except subprocess.CalledProcessError as e:
            logger.error(f"PostgreSQL backup failed: {str(e)}")
            raise

    def backup_vector_db(self, db_type: str) -> str:
        """
        Backup vector database.

        Args:
            db_type: Type of vector database (qdrant, milvus, chroma)

        Returns:
            Path to the backup directory
        """
        logger.info(f"Backing up {db_type} vector database...")

        # Get vector database configuration
        vector_config = self.config.get("vector_db", {}).get(db_type, {})

        # Create backup directory
        backup_dir = os.path.join(self.backup_dir, f"{db_type}_{self.timestamp}")
        os.makedirs(backup_dir, exist_ok=True)

        if db_type == "qdrant":
            # Backup Qdrant
            qdrant_path = vector_config.get("path", "/var/lib/qdrant")
            try:
                # Copy Qdrant data directory
                shutil.copytree(qdrant_path, os.path.join(backup_dir, "data"))
                logger.info(f"Qdrant backup completed: {backup_dir}")
                return backup_dir
            except Exception as e:
                logger.error(f"Qdrant backup failed: {str(e)}")
                raise

        elif db_type == "chroma":
            # Backup Chroma
            chroma_path = vector_config.get("path", "/var/lib/chroma")
            try:
                # Copy Chroma data directory
                shutil.copytree(chroma_path, os.path.join(backup_dir, "data"))
                logger.info(f"Chroma backup completed: {backup_dir}")
                return backup_dir
            except Exception as e:
                logger.error(f"Chroma backup failed: {str(e)}")
                raise

        elif db_type == "milvus":
            # Backup Milvus
            milvus_path = vector_config.get("path", "/var/lib/milvus")
            try:
                # Copy Milvus data directory
                shutil.copytree(milvus_path, os.path.join(backup_dir, "data"))
                logger.info(f"Milvus backup completed: {backup_dir}")
                return backup_dir
            except Exception as e:
                logger.error(f"Milvus backup failed: {str(e)}")
                raise
        else:
            logger.error(f"Unsupported vector database type: {db_type}")
            raise ValueError(f"Unsupported vector database type: {db_type}")

    def restore_postgresql(self, backup_file: str) -> bool:
        """
        Restore PostgreSQL database from backup.

        Args:
            backup_file: Path to the backup file

        Returns:
            True if restoration was successful
        """
        logger.info(f"Restoring PostgreSQL database from {backup_file}...")

        # Get PostgreSQL configuration
        pg_config = self.config.get("postgresql", {})
        db_name = pg_config.get("db_name", "deep_recall")
        db_user = pg_config.get("db_user", "postgres")
        db_host = pg_config.get("db_host", "localhost")
        db_port = pg_config.get("db_port", "5432")

        try:
            # Drop and recreate database
            drop_cmd = [
                "dropdb",
                "-h",
                db_host,
                "-p",
                db_port,
                "-U",
                db_user,
                "--if-exists",
                db_name,
            ]

            create_cmd = [
                "createdb",
                "-h",
                db_host,
                "-p",
                db_port,
                "-U",
                db_user,
                db_name,
            ]

            # Set PGPASSWORD environment variable if provided
            env = os.environ.copy()
            if "db_password" in pg_config:
                env["PGPASSWORD"] = pg_config["db_password"]

            # Drop and recreate database
            subprocess.run(drop_cmd, env=env, check=True)
            subprocess.run(create_cmd, env=env, check=True)

            # Restore from backup
            restore_cmd = [
                "psql",
                "-h",
                db_host,
                "-p",
                db_port,
                "-U",
                db_user,
                "-d",
                db_name,
                "-f",
                backup_file,
            ]

            subprocess.run(restore_cmd, env=env, check=True)
            logger.info(f"PostgreSQL restore completed from {backup_file}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"PostgreSQL restore failed: {str(e)}")
            return False

    def restore_vector_db(self, db_type: str, backup_dir: str) -> bool:
        """
        Restore vector database from backup.

        Args:
            db_type: Type of vector database (qdrant, milvus, chroma)
            backup_dir: Path to the backup directory

        Returns:
            True if restoration was successful
        """
        logger.info(f"Restoring {db_type} vector database from {backup_dir}...")

        # Get vector database configuration
        vector_config = self.config.get("vector_db", {}).get(db_type, {})

        if db_type == "qdrant":
            # Restore Qdrant
            qdrant_path = vector_config.get("path", "/var/lib/qdrant")
            try:
                # Stop Qdrant service if running
                subprocess.run(["systemctl", "stop", "qdrant"], check=False)

                # Remove existing data
                if os.path.exists(qdrant_path):
                    shutil.rmtree(qdrant_path)

                # Restore from backup
                shutil.copytree(os.path.join(backup_dir, "data"), qdrant_path)

                # Start Qdrant service
                subprocess.run(["systemctl", "start", "qdrant"], check=False)

                logger.info(f"Qdrant restore completed from {backup_dir}")
                return True
            except Exception as e:
                logger.error(f"Qdrant restore failed: {str(e)}")
                return False

        elif db_type == "chroma":
            # Restore Chroma
            chroma_path = vector_config.get("path", "/var/lib/chroma")
            try:
                # Stop Chroma service if running
                subprocess.run(["systemctl", "stop", "chroma"], check=False)

                # Remove existing data
                if os.path.exists(chroma_path):
                    shutil.rmtree(chroma_path)

                # Restore from backup
                shutil.copytree(os.path.join(backup_dir, "data"), chroma_path)

                # Start Chroma service
                subprocess.run(["systemctl", "start", "chroma"], check=False)

                logger.info(f"Chroma restore completed from {backup_dir}")
                return True
            except Exception as e:
                logger.error(f"Chroma restore failed: {str(e)}")
                return False

        elif db_type == "milvus":
            # Restore Milvus
            milvus_path = vector_config.get("path", "/var/lib/milvus")
            try:
                # Stop Milvus service if running
                subprocess.run(["systemctl", "stop", "milvus"], check=False)

                # Remove existing data
                if os.path.exists(milvus_path):
                    shutil.rmtree(milvus_path)

                # Restore from backup
                shutil.copytree(os.path.join(backup_dir, "data"), milvus_path)

                # Start Milvus service
                subprocess.run(["systemctl", "start", "milvus"], check=False)

                logger.info(f"Milvus restore completed from {backup_dir}")
                return True
            except Exception as e:
                logger.error(f"Milvus restore failed: {str(e)}")
                return False
        else:
            logger.error(f"Unsupported vector database type: {db_type}")
            return False

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List available backups.

        Returns:
            List of backup information
        """
        logger.info("Listing available backups...")

        backups = []

        # List PostgreSQL backups
        for file in os.listdir(self.backup_dir):
            if file.startswith("postgresql_") and file.endswith(".sql"):
                backup_path = os.path.join(self.backup_dir, file)
                backup_time = file.replace("postgresql_", "").replace(".sql", "")
                backups.append(
                    {
                        "type": "postgresql",
                        "path": backup_path,
                        "timestamp": backup_time,
                    }
                )

        # List vector database backups
        for dir_name in os.listdir(self.backup_dir):
            if dir_name.startswith(("qdrant_", "milvus_", "chroma_")):
                backup_path = os.path.join(self.backup_dir, dir_name)
                db_type = dir_name.split("_")[0]
                backup_time = dir_name.split("_")[1]
                backups.append(
                    {"type": db_type, "path": backup_path, "timestamp": backup_time}
                )

        # Sort backups by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)

        return backups


def main():
    """Main function to run backup and recovery operations."""
    parser = argparse.ArgumentParser(description="Backup and recovery utility")
    parser.add_argument(
        "--config",
        type=str,
        default="config/backup_config.json",
        help="Path to backup configuration file",
    )
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["backup", "restore", "list"],
        help="Action to perform",
    )
    parser.add_argument(
        "--db-type",
        type=str,
        choices=["postgresql", "qdrant", "milvus", "chroma", "all"],
        help="Database type to backup/restore",
    )
    parser.add_argument(
        "--backup-path", type=str, help="Path to backup file/directory for restore"
    )

    args = parser.parse_args()

    # Initialize backup manager
    backup_manager = BackupManager(args.config)

    if args.action == "backup":
        if args.db_type == "postgresql" or args.db_type == "all":
            backup_manager.backup_postgresql()

        if args.db_type == "qdrant" or args.db_type == "all":
            backup_manager.backup_vector_db("qdrant")

        if args.db_type == "milvus" or args.db_type == "all":
            backup_manager.backup_vector_db("milvus")

        if args.db_type == "chroma" or args.db_type == "all":
            backup_manager.backup_vector_db("chroma")

        if args.db_type is None:
            # Default to backing up all databases
            backup_manager.backup_postgresql()
            backup_manager.backup_vector_db("qdrant")
            backup_manager.backup_vector_db("milvus")
            backup_manager.backup_vector_db("chroma")

    elif args.action == "restore":
        if not args.backup_path:
            logger.error("Backup path is required for restore action")
            return

        if args.db_type == "postgresql":
            backup_manager.restore_postgresql(args.backup_path)

        elif args.db_type in ["qdrant", "milvus", "chroma"]:
            backup_manager.restore_vector_db(args.db_type, args.backup_path)

        else:
            logger.error("Database type is required for restore action")

    elif args.action == "list":
        backups = backup_manager.list_backups()

        print("\nAvailable Backups:")
        print("-----------------")

        for backup in backups:
            print(f"Type: {backup['type']}")
            print(f"Timestamp: {backup['timestamp']}")
            print(f"Path: {backup['path']}")
            print("-----------------")


if __name__ == "__main__":
    main()
