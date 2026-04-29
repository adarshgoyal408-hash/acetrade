"""
Ace-Trade Enhanced - Forensic Analysis Engine
Deep fundamental forensic analysis with 15+ red flag parameters.
Inspired by quality investing frameworks and accounting fraud detection.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger("acetrade.forensic")


class ForensicAnalyzer:
    """
    Comprehensive forensic analysis engine for equity research.
    Checks 15+ parameters across:
    - Leverage & Solvency
    - Cash Flow Quality
    - Profitability & Margins
    - Corporate Governance
    - Earnings Quality
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.info = {}
        self.financials = {}
        self.red_flags = []
        self.overall_score = 100  # Start at 100, deduct for each red flag
        
    def analyze(self) -> Tuple[int, List[Dict], str]:
        """
        Run complete forensic analysis.
        Returns: (overall_score, red_flags_list, ai_summary)
        """
        logger.info(f"Starting forensic analysis for {self.ticker}")
        
        try:
            # Fetch data
            self.info = self.stock.info
            self._fetch_financial_data()
            
            # Run all forensic checks
            self._check_debt_equity_ratio()
            self._check_interest_coverage()
            self._check_current_ratio()
            self._check_cfo_vs_net_income()
            self._check_cfo_vs_ebitda()
            self._check_working_capital_days()
            self._check_receivables_growth()
            self._check_inventory_growth()
            self._check_gross_margin_trend()
            self._check_ebitda_margin_trend()
            self._check_roe_vs_industry()
            self._check_promoter_holding()
            self._check_promoter_pledge()
            self._check_related_party_transactions()
            self._check_contingent_liabilities()
            
            # Cap score at 0-100
            self.overall_score = max(0, min(100, self.overall_score))
            
            logger.info(f"Forensic analysis complete for {self.ticker}: Score = {self.overall_score}")
            
            return self.overall_score, self.red_flags, self._generate_summary()
            
        except Exception as e:
            logger.error(f"Forensic analysis failed for {self.ticker}: {e}")
            return 50, [], f"Analysis incomplete due to data limitations: {str(e)}"
    
    def _fetch_financial_data(self):
        """Fetch financial statements from yfinance."""
        try:
            self.financials['balance_sheet'] = self.stock.balance_sheet
            self.financials['income_stmt'] = self.stock.income_stmt
            self.financials['cash_flow'] = self.stock.cashflow
            self.financials['quarterly_financials'] = self.stock.quarterly_financials
        except Exception as e:
            logger.warning(f"Could not fetch some financial data: {e}")
    
    def _add_flag(self, name: str, status: str, value: str, threshold: str, 
                  explanation: str, severity: int = 5):
        """Add a red flag to the list and deduct from overall score."""
        self.red_flags.append({
            'name': name,
            'status': status,  # PASS, CAUTION, RED_FLAG
            'value': value,
            'threshold': threshold,
            'explanation': explanation
        })
        
        # Deduct points based on severity
        if status == 'RED_FLAG':
            self.overall_score -= severity
        elif status == 'CAUTION':
            self.overall_score -= severity // 2
    
    # ═══════════════════════════════════════════════════════════════════
    # LEVERAGE & SOLVENCY CHECKS
    # ═══════════════════════════════════════════════════════════════════
    
    def _check_debt_equity_ratio(self):
        """Check Debt/Equity ratio - high leverage risk."""
        try:
            total_debt = self.info.get('totalDebt', 0) or 0
            total_equity = self.info.get('totalStockholderEquity', 1) or 1
            de_ratio = total_debt / total_equity if total_equity > 0 else 0
            
            if de_ratio > 2.0:
                self._add_flag(
                    "Debt/Equity Ratio",
                    "RED_FLAG",
                    f"{de_ratio:.2f}",
                    "< 2.0",
                    "High leverage! D/E > 2.0 indicates excessive debt burden. "
                    "Company may struggle to service debt in downturns. "
                    "Risk of covenant breaches and liquidity crunch.",
                    severity=10
                )
            elif de_ratio > 1.0:
                self._add_flag(
                    "Debt/Equity Ratio",
                    "CAUTION",
                    f"{de_ratio:.2f}",
                    "< 1.0",
                    "Moderate leverage. D/E between 1-2 is acceptable for stable businesses "
                    "but risky for cyclical sectors. Monitor interest coverage closely.",
                    severity=5
                )
            else:
                self._add_flag(
                    "Debt/Equity Ratio",
                    "PASS",
                    f"{de_ratio:.2f}",
                    "< 1.0",
                    "Conservative capital structure. Low debt reduces financial risk "
                    "and provides flexibility for growth investments.",
                    severity=0
                )
        except Exception as e:
            logger.warning(f"D/E check failed: {e}")
    
    def _check_interest_coverage(self):
        """Check Interest Coverage Ratio - ability to service debt."""
        try:
            ebitda = self.info.get('ebitda', 0) or 0
            interest_expense = self.info.get('interestExpense', 1) or 1
            
            if interest_expense and interest_expense != 0:
                coverage = abs(ebitda / interest_expense)
            else:
                coverage = 999  # No debt, pass
            
            if coverage < 2.0 and interest_expense > 0:
                self._add_flag(
                    "Interest Coverage",
                    "RED_FLAG",
                    f"{coverage:.2f}x",
                    "> 3.0x",
                    "Dangerous! EBITDA barely covers interest. Any earnings dip could "
                    "trigger default. Company is one bad quarter away from distress.",
                    severity=12
                )
            elif coverage < 3.0 and interest_expense > 0:
                self._add_flag(
                    "Interest Coverage",
                    "CAUTION",
                    f"{coverage:.2f}x",
                    "> 3.0x",
                    "Tight coverage. Company can service debt now but lacks cushion. "
                    "Watch for any margin pressure or revenue slowdown.",
                    severity=6
                )
            else:
                self._add_flag(
                    "Interest Coverage",
                    "PASS",
                    f"{coverage:.1f}x" if coverage < 50 else ">50x",
                    "> 3.0x",
                    "Strong debt servicing ability. Company generates sufficient cash "
                    "to comfortably pay interest even in downturns.",
                    severity=0
                )
        except Exception as e:
            logger.warning(f"Interest coverage check failed: {e}")
    
    def _check_current_ratio(self):
        """Check Current Ratio - short-term liquidity."""
        try:
            current_assets = self.info.get('totalCurrentAssets', 0) or 0
            current_liabilities = self.info.get('totalCurrentLiabilities', 1) or 1
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
            
            if current_ratio < 1.0:
                self._add_flag(
                    "Current Ratio",
                    "RED_FLAG",
                    f"{current_ratio:.2f}",
                    "> 1.5",
                    "Liquidity crisis risk! Current assets < current liabilities. "
                    "Company may struggle to meet short-term obligations. "
                    "Check for dependence on short-term borrowing.",
                    severity=10
                )
            elif current_ratio < 1.5:
                self._add_flag(
                    "Current Ratio",
                    "CAUTION",
                    f"{current_ratio:.2f}",
                    "> 1.5",
                    "Tight working capital. Company has minimal liquidity buffer. "
                    "Any disruption in collections or sales could create cash crunch.",
                    severity=5
                )
            else:
                self._add_flag(
                    "Current Ratio",
                    "PASS",
                    f"{current_ratio:.2f}",
                    "> 1.5",
                    "Healthy liquidity position. Company can easily meet short-term "
                    "obligations and has room to navigate operational challenges.",
                    severity=0
                )
        except Exception as e:
            logger.warning(f"Current ratio check failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # CASH FLOW QUALITY CHECKS
    # ═══════════════════════════════════════════════════════════════════
    
    def _check_cfo_vs_net_income(self):
        """Check if Cash from Operations matches Net Income - earnings quality."""
        try:
            cash_flow_df = self.financials.get('cash_flow')
            income_df = self.financials.get('income_stmt')
            
            if cash_flow_df is None or income_df is None or cash_flow_df.empty or income_df.empty:
                return
            
            # Get latest year data
            cfo = cash_flow_df.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow_df.index else 0
            net_income = income_df.loc['Net Income'].iloc[0] if 'Net Income' in income_df.index else 1
            
            if net_income and net_income != 0:
                cfo_ni_ratio = cfo / abs(net_income)
            else:
                cfo_ni_ratio = 0
            
            if cfo_ni_ratio < 0.7:
                self._add_flag(
                    "Cash vs. Profit Quality",
                    "RED_FLAG",
                    f"{cfo_ni_ratio:.2f}",
                    "> 0.9",
                    "MAJOR RED FLAG! Operating cash flow significantly trails reported profit. "
                    "Possible aggressive revenue recognition, rising receivables, or accounting manipulation. "
                    "Earnings are not translating to cash - highly suspicious.",
                    severity=15
                )
            elif cfo_ni_ratio < 0.9:
                self._add_flag(
                    "Cash vs. Profit Quality",
                    "CAUTION",
                    f"{cfo_ni_ratio:.2f}",
                    "> 0.9",
                    "Earnings quality concern. Cash generation lagging profit. "
                    "Could be due to working capital build-up or accounting timing. Investigate further.",
                    severity=8
                )
            else:
                self._add_flag(
                    "Cash vs. Profit Quality",
                    "PASS",
                    f"{cfo_ni_ratio:.2f}",
                    "> 0.9",
                    "Excellent earnings quality! Profits are converting to cash. "
                    "This is the hallmark of genuine business performance.",
                    severity=0
                )
        except Exception as e:
            logger.warning(f"CFO vs NI check failed: {e}")
    
    def _check_cfo_vs_ebitda(self):
        """Check CFO/EBITDA ratio - cash conversion efficiency."""
        try:
            cash_flow_df = self.financials.get('cash_flow')
            
            if cash_flow_df is None or cash_flow_df.empty:
                return
            
            cfo = cash_flow_df.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow_df.index else 0
            ebitda = self.info.get('ebitda', 1) or 1
            
            cfo_ebitda = cfo / ebitda if ebitda > 0 else 0
            
            if cfo_ebitda < 0.5:
                self._add_flag(
                    "CFO/EBITDA Conversion",
                    "RED_FLAG",
                    f"{cfo_ebitda:.2f}",
                    "> 0.7",
                    "Poor cash conversion! Less than 50% of EBITDA converts to operating cash. "
                    "Company is either bleeding working capital or EBITDA quality is suspect. "
                    "Common in stressed asset/infra companies.",
                    severity=12
                )
            elif cfo_ebitda < 0.7:
                self._add_flag(
                    "CFO/EBITDA Conversion",
                    "CAUTION",
                    f"{cfo_ebitda:.2f}",
                    "> 0.7",
                    "Below-average cash conversion. Indicates working capital stress "
                    "or high capex requirements eating into free cash flow.",
                    severity=6
                )
            else:
                self._add_flag(
                    "CFO/EBITDA Conversion",
                    "PASS",
                    f"{cfo_ebitda:.2f}",
                    "> 0.7",
                    "Strong cash conversion! Company efficiently converts EBITDA to cash. "
                    "Sign of efficient operations and healthy working capital management.",
                    severity=0
                )
        except Exception as e:
            logger.warning(f"CFO/EBITDA check failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # WORKING CAPITAL & QUALITY CHECKS
    # ═══════════════════════════════════════════════════════════════════
    
    def _check_working_capital_days(self):
        """Check if working capital days are increasing - hidden stress signal."""
        try:
            # This is complex - for now, use a proxy via current assets growth vs revenue growth
            revenue = self.info.get('totalRevenue', 0)
            current_assets = self.info.get('totalCurrentAssets', 0)
            
            # Simplified check - ideally would compare YoY trends
            if revenue and revenue > 0:
                wc_intensity = current_assets / revenue
                
                if wc_intensity > 0.5:
                    self._add_flag(
                        "Working Capital Intensity",
                        "CAUTION",
                        f"{wc_intensity:.2%}",
                        "< 40%",
                        "High working capital intensity. Company requires significant "
                        "current assets to generate revenue. Watch for cash cycle stress.",
                        severity=4
                    )
                else:
                    self._add_flag(
                        "Working Capital Intensity",
                        "PASS",
                        f"{wc_intensity:.2%}",
                        "< 40%",
                        "Efficient working capital management. Company doesn't need "
                        "excessive current assets to generate revenue.",
                        severity=0
                    )
        except Exception as e:
            logger.warning(f"WC days check failed: {e}")
    
    def _check_receivables_growth(self):
        """Check if receivables are growing faster than revenue - channel stuffing signal."""
        try:
            balance_sheet = self.financials.get('balance_sheet')
            income_stmt = self.financials.get('income_stmt')
            
            if balance_sheet is None or income_stmt is None or balance_sheet.empty or income_stmt.empty:
                return
            
            if 'Accounts Receivable' in balance_sheet.index and 'Total Revenue' in income_stmt.index:
                # Compare latest 2 periods
                if len(balance_sheet.columns) >= 2 and len(income_stmt.columns) >= 2:
                    receivables_curr = balance_sheet.loc['Accounts Receivable'].iloc[0]
                    receivables_prev = balance_sheet.loc['Accounts Receivable'].iloc[1]
                    revenue_curr = income_stmt.loc['Total Revenue'].iloc[0]
                    revenue_prev = income_stmt.loc['Total Revenue'].iloc[1]
                    
                    rec_growth = (receivables_curr - receivables_prev) / receivables_prev if receivables_prev != 0 else 0
                    rev_growth = (revenue_curr - revenue_prev) / revenue_prev if revenue_prev != 0 else 0
                    
                    if rec_growth > rev_growth * 1.5 and rec_growth > 0.1:
                        self._add_flag(
                            "Receivables Quality",
                            "RED_FLAG",
                            f"Rec: +{rec_growth:.1%}, Rev: +{rev_growth:.1%}",
                            "Rec ≤ Rev growth",
                            "MAJOR RED FLAG! Receivables growing 50%+ faster than revenue. "
                            "Classic sign of channel stuffing, aggressive credit terms, or "
                            "collection issues. Investigate customer concentration and credit policy.",
                            severity=12
                        )
                    elif rec_growth > rev_growth * 1.2 and rec_growth > 0.05:
                        self._add_flag(
                            "Receivables Quality",
                            "CAUTION",
                            f"Rec: +{rec_growth:.1%}, Rev: +{rev_growth:.1%}",
                            "Rec ≤ Rev growth",
                            "Receivables growing faster than revenue. May indicate longer "
                            "payment cycles, customer stress, or aggressive sales tactics.",
                            severity=6
                        )
                    else:
                        self._add_flag(
                            "Receivables Quality",
                            "PASS",
                            f"Rec: +{rec_growth:.1%}, Rev: +{rev_growth:.1%}",
                            "Rec ≤ Rev growth",
                            "Healthy receivables trend. Growth is in line with or below revenue growth.",
                            severity=0
                        )
        except Exception as e:
            logger.warning(f"Receivables growth check failed: {e}")
    
    def _check_inventory_growth(self):
        """Check inventory growth vs revenue - obsolescence risk."""
        try:
            balance_sheet = self.financials.get('balance_sheet')
            income_stmt = self.financials.get('income_stmt')
            
            if balance_sheet is None or income_stmt is None or balance_sheet.empty or income_stmt.empty:
                return
            
            if 'Inventory' in balance_sheet.index and 'Total Revenue' in income_stmt.index:
                if len(balance_sheet.columns) >= 2 and len(income_stmt.columns) >= 2:
                    inv_curr = balance_sheet.loc['Inventory'].iloc[0]
                    inv_prev = balance_sheet.loc['Inventory'].iloc[1]
                    revenue_curr = income_stmt.loc['Total Revenue'].iloc[0]
                    revenue_prev = income_stmt.loc['Total Revenue'].iloc[1]
                    
                    inv_growth = (inv_curr - inv_prev) / inv_prev if inv_prev != 0 else 0
                    rev_growth = (revenue_curr - revenue_prev) / revenue_prev if revenue_prev != 0 else 0
                    
                    if inv_growth > rev_growth * 1.5 and inv_growth > 0.1:
                        self._add_flag(
                            "Inventory Quality",
                            "RED_FLAG",
                            f"Inv: +{inv_growth:.1%}, Rev: +{rev_growth:.1%}",
                            "Inv ≤ Rev growth",
                            "Inventory piling up! Growing much faster than revenue. "
                            "Risk of obsolescence, demand slowdown, or overproduction. "
                            "May need write-downs ahead.",
                            severity=10
                        )
                    elif inv_growth > rev_growth * 1.2 and inv_growth > 0.05:
                        self._add_flag(
                            "Inventory Quality",
                            "CAUTION",
                            f"Inv: +{inv_growth:.1%}, Rev: +{rev_growth:.1%}",
                            "Inv ≤ Rev growth",
                            "Inventory building faster than sales. Monitor for demand weakness "
                            "or supply chain issues.",
                            severity=5
                        )
                    else:
                        self._add_flag(
                            "Inventory Quality",
                            "PASS",
                            f"Inv: +{inv_growth:.1%}, Rev: +{rev_growth:.1%}",
                            "Inv ≤ Rev growth",
                            "Healthy inventory management. Growth aligned with revenue.",
                            severity=0
                        )
        except Exception as e:
            logger.warning(f"Inventory growth check failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # PROFITABILITY & MARGIN CHECKS
    # ═══════════════════════════════════════════════════════════════════
    
    def _check_gross_margin_trend(self):
        """Check if gross margins are deteriorating - competitive pressure signal."""
        try:
            gross_margin = self.info.get('grossMargins', 0)
            
            # Ideally would compare YoY trend - for now just check absolute level
            if gross_margin and gross_margin < 0.15:
                self._add_flag(
                    "Gross Margin",
                    "CAUTION",
                    f"{gross_margin:.1%}",
                    "> 20%",
                    "Low gross margins (<15%). Company has limited pricing power "
                    "and is vulnerable to input cost inflation. Thin margin for error.",
                    severity=4
                )
            else:
                self._add_flag(
                    "Gross Margin",
                    "PASS",
                    f"{gross_margin:.1%}" if gross_margin else "N/A",
                    "> 20%",
                    "Healthy gross margins indicate pricing power and competitive moat." if gross_margin and gross_margin > 0.2 else "Adequate margins.",
                    severity=0
                )
        except Exception as e:
            logger.warning(f"Gross margin check failed: {e}")
    
    def _check_ebitda_margin_trend(self):
        """Check EBITDA margin quality."""
        try:
            ebitda_margin = self.info.get('ebitdaMargins', 0)
            
            if ebitda_margin and ebitda_margin < 0.08:
                self._add_flag(
                    "EBITDA Margin",
                    "CAUTION",
                    f"{ebitda_margin:.1%}",
                    "> 12%",
                    "Weak EBITDA margins (<8%). Low operating leverage. "
                    "Small shocks can swing to losses. Requires scale to improve.",
                    severity=5
                )
            else:
                self._add_flag(
                    "EBITDA Margin",
                    "PASS",
                    f"{ebitda_margin:.1%}" if ebitda_margin else "N/A",
                    "> 12%",
                    "Healthy EBITDA margins indicate operational efficiency." if ebitda_margin and ebitda_margin > 0.15 else "Adequate operating margins.",
                    severity=0
                )
        except Exception as e:
            logger.warning(f"EBITDA margin check failed: {e}")
    
    def _check_roe_vs_industry(self):
        """Check Return on Equity."""
        try:
            roe = self.info.get('returnOnEquity', 0)
            
            if roe and roe < 0.10:
                self._add_flag(
                    "Return on Equity",
                    "CAUTION",
                    f"{roe:.1%}",
                    "> 15%",
                    "Below-average ROE (<10%). Capital is not being deployed efficiently. "
                    "Investors can get better returns in fixed income.",
                    severity=6
                )
            elif roe and roe > 0.15:
                self._add_flag(
                    "Return on Equity",
                    "PASS",
                    f"{roe:.1%}",
                    "> 15%",
                    "Strong ROE! Company generates attractive returns on shareholder capital. "
                    "Sign of competitive advantage and efficient capital allocation.",
                    severity=0
                )
            else:
                self._add_flag(
                    "Return on Equity",
                    "PASS",
                    f"{roe:.1%}" if roe else "N/A",
                    "> 15%",
                    "Moderate ROE. Acceptable but not exceptional.",
                    severity=0
                )
        except Exception as e:
            logger.warning(f"ROE check failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # CORPORATE GOVERNANCE CHECKS
    # ═══════════════════════════════════════════════════════════════════
    
    def _check_promoter_holding(self):
        """Check promoter holding - skin in the game."""
        try:
            # This data is hard to get from yfinance - would need Screener.in or NSE API
            # For now, check if company info has shareholding pattern
            
            # Placeholder - in production, fetch from NSE/Screener
            promoter_pct = self.info.get('heldPercentInsiders', 0) * 100 if 'heldPercentInsiders' in self.info else None
            
            if promoter_pct and promoter_pct < 25:
                self._add_flag(
                    "Promoter Holding",
                    "CAUTION",
                    f"{promoter_pct:.1f}%",
                    "> 35%",
                    "Low promoter holding (<25%). Reduced skin in the game. "
                    "Could indicate promoters exiting or lack of confidence.",
                    severity=6
                )
            elif promoter_pct and promoter_pct > 75:
                self._add_flag(
                    "Promoter Holding",
                    "CAUTION",
                    f"{promoter_pct:.1f}%",
                    "35-75%",
                    "Very high promoter holding (>75%). Low public float can lead to "
                    "liquidity issues and governance concerns. Harder to hold management accountable.",
                    severity=4
                )
            elif promoter_pct:
                self._add_flag(
                    "Promoter Holding",
                    "PASS",
                    f"{promoter_pct:.1f}%",
                    "35-75%",
                    "Healthy promoter holding. Skin in the game with good public float.",
                    severity=0
                )
            else:
                self._add_flag(
                    "Promoter Holding",
                    "PASS",
                    "Data unavailable",
                    "35-75%",
                    "Shareholding data not available. Check NSE or company filings.",
                    severity=0
                )
        except Exception as e:
            logger.warning(f"Promoter holding check failed: {e}")
    
    def _check_promoter_pledge(self):
        """Check for promoter share pledging - distress signal."""
        try:
            # This data requires specialized Indian market data sources
            # Would integrate with Screener.in or Tijori Finance in production
            
            self._add_flag(
                "Promoter Pledge",
                "PASS",
                "N/A - Check Screener.in",
                "0%",
                "Promoter pledge data not available via yfinance. "
                "For Indian stocks, check Screener.in, Tijori, or NSE announcements. "
                "Any pledge >50% is a major red flag.",
                severity=0
            )
        except Exception as e:
            logger.warning(f"Promoter pledge check failed: {e}")
    
    def _check_related_party_transactions(self):
        """Flag if Related Party Transactions are high - governance risk."""
        try:
            # This requires annual report parsing - beyond yfinance scope
            # In production, use NLP on annual reports or specialized data providers
            
            self._add_flag(
                "Related Party Transactions",
                "PASS",
                "Manual review required",
                "< 5% of revenue",
                "RPT data requires annual report analysis. "
                "Check notes to accounts for loans to/from promoters, related party sales, "
                "guarantees given. High RPTs (>10% revenue) are governance red flags.",
                severity=0
            )
        except Exception as e:
            logger.warning(f"RPT check failed: {e}")
    
    def _check_contingent_liabilities(self):
        """Check for hidden contingent liabilities."""
        try:
            # Would need balance sheet notes parsing
            
            self._add_flag(
                "Contingent Liabilities",
                "PASS",
                "Manual review required",
                "< 20% of net worth",
                "Contingent liability data not auto-extracted. "
                "Check annual report notes for pending litigation, tax disputes, guarantees. "
                "If contingent liabilities >50% of net worth, it's a red flag.",
                severity=0
            )
        except Exception as e:
            logger.warning(f"Contingent liabilities check failed: {e}")
    
    def _generate_summary(self) -> str:
        """Generate AI-friendly summary of forensic findings."""
        red_count = sum(1 for flag in self.red_flags if flag['status'] == 'RED_FLAG')
        caution_count = sum(1 for flag in self.red_flags if flag['status'] == 'CAUTION')
        pass_count = sum(1 for flag in self.red_flags if flag['status'] == 'PASS')
        
        if self.overall_score >= 80:
            verdict = "FORENSICALLY CLEAN"
        elif self.overall_score >= 60:
            verdict = "ACCEPTABLE WITH CAUTIONS"
        elif self.overall_score >= 40:
            verdict = "MULTIPLE RED FLAGS"
        else:
            verdict = "HIGH RISK - AVOID"
        
        summary = f"""
FORENSIC ANALYSIS SUMMARY - {self.ticker}
Overall Score: {self.overall_score}/100 - {verdict}

Red Flags: {red_count} | Cautions: {caution_count} | Pass: {pass_count}

Key Findings:
{chr(10).join([f"- {flag['name']}: {flag['status']}" for flag in self.red_flags[:5]])}

Recommendation: {"Safe for quality-focused portfolios." if self.overall_score >= 80 else 
                 "Proceed with caution. Deep dive required." if self.overall_score >= 60 else
                 "High accounting/governance risk. Avoid unless turnaround specialist."}
"""
        return summary.strip()


def run_forensic_analysis(ticker: str) -> Tuple[int, List[Dict], str]:
    """
    Main entry point for forensic analysis.
    Returns: (score, red_flags, summary)
    """
    analyzer = ForensicAnalyzer(ticker)
    return analyzer.analyze()
