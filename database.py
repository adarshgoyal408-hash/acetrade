"""
Ace-Trade Enhanced - SQLite Database Layer
Handles persistent storage for equity research reports, forensic analysis, and historical data.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("acetrade.db")

DB_PATH = "acetrade_research.db"


def init_database():
    """Initialize database schema. Call once at app startup."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Research Reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            report_date TEXT NOT NULL,
            analyst_name TEXT DEFAULT 'Ace-Trade AI',
            
            -- Stock Info
            sector TEXT,
            industry TEXT,
            market_cap_category TEXT,
            market_cap_crore REAL,
            current_price REAL,
            exchange TEXT,
            
            -- Executive Summary
            rating TEXT,
            target_price REAL,
            upside_potential REAL,
            investment_thesis TEXT,
            key_catalysts TEXT,
            key_risks TEXT,
            
            -- Financial Data (JSON)
            financial_data TEXT,
            
            -- Forensic Analysis
            forensic_score INTEGER,
            red_flags TEXT,
            
            -- Metadata
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(ticker, report_date)
        )
    """)
    
    # Forensic Analysis History
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forensic_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            analysis_date TEXT NOT NULL,
            overall_score INTEGER,
            parameters TEXT,
            ai_summary TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(ticker, analysis_date)
        )
    """)
    
    # Report Generation History
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS report_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            report_id INTEGER,
            action TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (report_id) REFERENCES research_reports(id)
        )
    """)
    
    # Ratings History
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ratings_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            rating TEXT NOT NULL,
            target_price REAL,
            current_price REAL,
            date TEXT NOT NULL,
            analyst TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def save_research_report(report_data: Dict) -> int:
    """Save a complete research report to database. Returns report ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO research_reports (
                ticker, company_name, report_date, analyst_name,
                sector, industry, market_cap_category, market_cap_crore,
                current_price, exchange, rating, target_price, upside_potential,
                investment_thesis, key_catalysts, key_risks,
                financial_data, forensic_score, red_flags, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_data['ticker'],
            report_data['company_name'],
            report_data['report_date'],
            report_data.get('analyst_name', 'Ace-Trade AI'),
            report_data.get('sector'),
            report_data.get('industry'),
            report_data.get('market_cap_category'),
            report_data.get('market_cap_crore'),
            report_data.get('current_price'),
            report_data.get('exchange'),
            report_data.get('rating'),
            report_data.get('target_price'),
            report_data.get('upside_potential'),
            report_data.get('investment_thesis'),
            report_data.get('key_catalysts'),
            report_data.get('key_risks'),
            json.dumps(report_data.get('financial_data', {})),
            report_data.get('forensic_score'),
            json.dumps(report_data.get('red_flags', [])),
            datetime.now().isoformat()
        ))
        
        report_id = cursor.lastrowid
        conn.commit()
        
        # Log to history
        cursor.execute("""
            INSERT INTO report_history (ticker, report_id, action)
            VALUES (?, ?, ?)
        """, (report_data['ticker'], report_id, 'CREATED'))
        conn.commit()
        
        logger.info(f"Saved research report for {report_data['ticker']} (ID: {report_id})")
        return report_id
        
    except Exception as e:
        logger.error(f"Error saving research report: {e}")
        conn.rollback()
        return -1
    finally:
        conn.close()


def get_research_report(ticker: str, date: Optional[str] = None) -> Optional[Dict]:
    """Get research report for ticker. If date is None, gets latest."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        if date:
            cursor.execute("""
                SELECT * FROM research_reports 
                WHERE ticker = ? AND report_date = ?
            """, (ticker, date))
        else:
            cursor.execute("""
                SELECT * FROM research_reports 
                WHERE ticker = ? 
                ORDER BY created_at DESC LIMIT 1
            """, (ticker,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        columns = [desc[0] for desc in cursor.description]
        report = dict(zip(columns, row))
        
        # Parse JSON fields
        report['financial_data'] = json.loads(report['financial_data']) if report['financial_data'] else {}
        report['red_flags'] = json.loads(report['red_flags']) if report['red_flags'] else []
        
        return report
        
    except Exception as e:
        logger.error(f"Error retrieving research report: {e}")
        return None
    finally:
        conn.close()


def list_all_reports(limit: int = 50) -> List[Dict]:
    """Get list of all research reports (summary view)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT ticker, company_name, report_date, rating, 
                   target_price, current_price, forensic_score, created_at
            FROM research_reports 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        reports = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return reports
        
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        return []
    finally:
        conn.close()


def save_forensic_analysis(ticker: str, score: int, parameters: List[Dict], ai_summary: str):
    """Save forensic analysis results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        analysis_date = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("""
            INSERT OR REPLACE INTO forensic_analysis 
            (ticker, analysis_date, overall_score, parameters, ai_summary)
            VALUES (?, ?, ?, ?, ?)
        """, (ticker, analysis_date, score, json.dumps(parameters), ai_summary))
        
        conn.commit()
        logger.info(f"Saved forensic analysis for {ticker}")
        
    except Exception as e:
        logger.error(f"Error saving forensic analysis: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_forensic_analysis(ticker: str) -> Optional[Dict]:
    """Get latest forensic analysis for ticker."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT * FROM forensic_analysis 
            WHERE ticker = ? 
            ORDER BY created_at DESC LIMIT 1
        """, (ticker,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        columns = [desc[0] for desc in cursor.description]
        analysis = dict(zip(columns, row))
        analysis['parameters'] = json.loads(analysis['parameters']) if analysis['parameters'] else []
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error retrieving forensic analysis: {e}")
        return None
    finally:
        conn.close()


def save_rating_history(ticker: str, rating: str, target_price: float, 
                       current_price: float, analyst: str = "Ace-Trade AI"):
    """Save rating to history for tracking."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO ratings_history 
            (ticker, rating, target_price, current_price, date, analyst)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, rating, target_price, current_price, 
              datetime.now().strftime("%Y-%m-%d"), analyst))
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error saving rating history: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_rating_history(ticker: str, limit: int = 10) -> List[Dict]:
    """Get rating history for a ticker."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT * FROM ratings_history 
            WHERE ticker = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (ticker, limit))
        
        columns = [desc[0] for desc in cursor.description]
        history = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return history
        
    except Exception as e:
        logger.error(f"Error retrieving rating history: {e}")
        return []
    finally:
        conn.close()


def delete_report(report_id: int) -> bool:
    """Delete a research report by ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM research_reports WHERE id = ?", (report_id,))
        conn.commit()
        
        # Log deletion
        cursor.execute("""
            INSERT INTO report_history (ticker, report_id, action)
            VALUES ('DELETED', ?, 'DELETED')
        """, (report_id,))
        conn.commit()
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting report: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_database_stats() -> Dict:
    """Get database statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM research_reports")
        stats['total_reports'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM forensic_analysis")
        stats['total_forensic_analyses'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ratings_history")
        stats['total_ratings'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT rating, COUNT(*) 
            FROM research_reports 
            GROUP BY rating
        """)
        stats['rating_distribution'] = dict(cursor.fetchall())
        
        cursor.execute("""
            SELECT ticker, COUNT(*) as count
            FROM research_reports 
            GROUP BY ticker 
            ORDER BY count DESC 
            LIMIT 10
        """)
        stats['most_analyzed_stocks'] = [
            {'ticker': row[0], 'count': row[1]} 
            for row in cursor.fetchall()
        ]
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {}
    finally:
        conn.close()
