# -*- coding: utf-8 -*-
"""
Prepare data for training.
Modified from OpenPipe/ART
"""

import logging
import os
import sqlite3
from datetime import datetime
from datasets import Dataset, Features, Sequence, Value, load_dataset
from tqdm import tqdm


# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Database will live in "../data/enron_emails.db" relative to project root
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "..", "..", "data", "enron_emails.db")

DEFAULT_REPO_ID = "corbt/enron-emails"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# --- Database Schema ---
SQL_CREATE_TABLES = """
DROP TABLE IF EXISTS recipients;
DROP TABLE IF EXISTS emails_fts;
DROP TABLE IF EXISTS emails;

CREATE TABLE emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE,
    subject TEXT,
    from_address TEXT,
    date TEXT, -- Store as ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    body TEXT,
    file_name TEXT
);

CREATE TABLE recipients (
    email_id INTEGER,
    recipient_address TEXT,
    recipient_type TEXT, -- 'to', 'cc', 'bcc'
    FOREIGN KEY(email_id) REFERENCES emails(id) ON DELETE CASCADE
);
"""

SQL_CREATE_INDEXES_TRIGGERS = """
CREATE INDEX idx_emails_from ON emails(from_address);
CREATE INDEX idx_emails_date ON emails(date);
CREATE INDEX idx_emails_message_id ON emails(message_id);
CREATE INDEX idx_recipients_address ON recipients(recipient_address);
CREATE INDEX idx_recipients_type ON recipients(recipient_type);
CREATE INDEX idx_recipients_email_id ON recipients(email_id);
CREATE INDEX idx_recipients_address_email ON recipients(
    recipient_address, email_id
);

CREATE VIRTUAL TABLE emails_fts USING fts5(
    subject,
    body,
    content='emails',
    content_rowid='id'
);

CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
    INSERT INTO emails_fts (rowid, subject, body)
    VALUES (new.id, new.subject, new.body);
END;

CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
    DELETE FROM emails_fts WHERE rowid=old.id;
END;

CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
    UPDATE emails_fts SET subject=new.subject, body=new.body
    WHERE rowid=old.id;
END;

INSERT INTO emails_fts (rowid, subject, body)
SELECT id, subject, body FROM emails;
"""


# --- Functions ---


def download_dataset(repo_id: str) -> Dataset:
    """Downloads the dataset from Hugging Face Hub."""
    logging.info(
        "Attempting to download dataset from Hugging Face Hub: %s",
        repo_id,
    )
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        },
    )
    dataset_obj = load_dataset(
        repo_id,
        features=expected_features,
        split="train",
    )
    # Basic type check remains useful
    if not isinstance(dataset_obj, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset_obj)}")
    logging.info(
        "Successfully loaded dataset '%s' with %d records.",
        repo_id,
        len(dataset_obj),
    )
    return dataset_obj


def create_database(db_path: str) -> None:
    """Creates the SQLite database and tables."""
    logging.info("Creating SQLite database and tables at: %s", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()
    conn.close()
    logging.info("Database tables created successfully.")


def _should_skip_email(
    body: str,
    message_id: str,
    to_list: list[str],
    cc_list: list[str],
    bcc_list: list[str],
) -> bool:
    """Check if email should be skipped based on filters."""
    if len(body) > 5000:
        logging.debug(
            "Skipping email %s: Body length > 5000 characters.",
            message_id,
        )
        return True

    total_recipients = len(to_list) + len(cc_list) + len(bcc_list)
    if total_recipients > 30:
        logging.debug(
            "Skipping email %s: Total recipients (%d) > 30.",
            message_id,
            total_recipients,
        )
        return True
    return False


def _prepare_recipient_data(
    email_pk_id: int,
    to_list: list[str],
    cc_list: list[str],
    bcc_list: list[str],
) -> list[tuple[int, str, str]]:
    """Prepare recipient data for database insertion."""
    recipient_data = []
    for addr in to_list:
        recipient_data.append((email_pk_id, addr, "to"))
    for addr in cc_list:
        recipient_data.append((email_pk_id, addr, "cc"))
    for addr in bcc_list:
        recipient_data.append((email_pk_id, addr, "bcc"))
    return recipient_data


def populate_database(db_path: str, dataset: Dataset) -> None:
    """Populates the database with data from the Hugging Face dataset."""
    logging.info("Populating database %s...", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Performance Pragmas ---
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")

    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_emails = set()

    conn.execute("BEGIN TRANSACTION;")

    for email_data in tqdm(dataset, desc="Inserting emails"):
        assert isinstance(email_data, dict)
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj: datetime = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]
        to_list_raw = email_data["to"]
        cc_list_raw = email_data["cc"]
        bcc_list_raw = email_data["bcc"]

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        to_list = [str(addr) for addr in to_list_raw if addr]
        cc_list = [str(addr) for addr in cc_list_raw if addr]
        bcc_list = [str(addr) for addr in bcc_list_raw if addr]

        if _should_skip_email(body, message_id, to_list, cc_list, bcc_list):
            skipped_count += 1
            continue

        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            logging.debug(
                "Skipping duplicate email (Subject: %s..., From: %s)",
                subject[:50],
                from_address,
            )
            duplicate_count += 1
            continue
        processed_emails.add(email_key)

        cursor.execute(
            """
            INSERT INTO emails (
                message_id, subject, from_address, date, body, file_name
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, subject, from_address, date_str, body, file_name),
        )
        email_pk_id = cursor.lastrowid
        if email_pk_id is None:
            logging.warning(
                "Failed to get email ID after insert for message_id: %s",
                message_id,
            )
            continue

        recipient_data = _prepare_recipient_data(
            email_pk_id,
            to_list,
            cc_list,
            bcc_list,
        )

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (
                    email_id, recipient_address, recipient_type
                )
                VALUES (?, ?, ?)
                """,
                recipient_data,
            )
        record_count += 1

    conn.commit()
    conn.close()
    logging.info("Successfully inserted %d email records.", record_count)
    if skipped_count > 0:
        logging.info(
            "Skipped %d email records due to length or recipient limits.",
            skipped_count,
        )
    if duplicate_count > 0:
        logging.info(
            "Skipped %d duplicate email records "
            "(based on subject, body, from).",
            duplicate_count,
        )


def create_indexes_and_triggers(db_path: str) -> None:
    """Creates indexes and triggers on the populated database."""
    logging.info("Creating indexes and triggers for database: %s...", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    conn.commit()
    conn.close()
    logging.info("Indexes and triggers created successfully.")


def generate_database(
    repo_id: str = DEFAULT_REPO_ID,
    db_path: str = DEFAULT_DB_PATH,
    overwrite: bool = False,
) -> None:
    """
    Generates the SQLite database from the specified Hugging Face dataset.
    Simplified version without extensive error handling.

    Args:
        repo_id: The Hugging Face repository ID for the dataset.
        db_path: The path where the SQLite database file should be
            created.
        overwrite: If True, any existing database file at db_path will
            be removed.
    """
    logging.info(
        "Starting database generation for repo '%s' at '%s'",
        repo_id,
        db_path,
    )
    logging.info("Overwrite existing database: %s", overwrite)

    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        logging.info("Creating data directory: %s", db_dir)
        os.makedirs(db_dir)

    if overwrite and os.path.exists(db_path):
        logging.warning("Removing existing database file: %s", db_path)
        os.remove(db_path)
    elif not overwrite and os.path.exists(db_path):
        # If not overwriting and file exists, subsequent steps might fail
        # or behave unexpectedly. We are removing the explicit error here
        # as requested.
        logging.warning(
            "Database file %s exists and overwrite is False. "
            "Assuming file is already generated.",
            db_path,
        )
        return

    # 1. Download dataset
    dataset = download_dataset(repo_id)

    # 2. Create database schema (Tables only)
    # Note: This will fail if overwrite=False and the file exists with
    # incompatible schema/data.
    create_database(db_path)

    # 3. Populate database
    populate_database(db_path, dataset)

    # 4. Create Indexes and Triggers
    create_indexes_and_triggers(db_path)

    logging.info("Database generation process completed for %s.", db_path)
    logging.info(
        "Please set the environment variable DEFAULT_EMAIL_DB_PATH "
        "to this path.",
    )


if __name__ == "__main__":
    generate_database(overwrite=True)
