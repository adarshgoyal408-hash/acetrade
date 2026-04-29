"""
Ace-Trade Enhanced - Equity Research Report Generator
Generates comprehensive equity research reports with AI-powered insights.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from forensic_analysis import run_forensic_analysis

logger = logging.getLogger("acetrade.research")


class ResearchReportGenerator:
    """
    Generates comprehensive equity research reports combining:
    - Stock fundamentals
    - Financial analysis
    - Forensic checks
    - Valuation
    - AI-powered insights
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(ticker)
        self.info = {}
        self.report_data = {}
        
    def generate_report(self, analyst_name: str = "Ace-Trade AI") -> Dict:
        """
        Generate complete research report.
        Returns dictionary with all report sections.
        """
        logger.info(f"Generating research report for {self.ticker}")
        
        try:
            # Fetch basic stock info
            self.info = self.stock.info
            
            # Build report sections
            self.report_data = {
                'ticker': self.ticker,
                'company_name': self._get_company_name(),
                'report_date': datetime.now().strftime("%Y-%m-%d"),
                'analyst_name': analyst_name,
                'last_updated': datetime.now().isoformat(),
                
                # Stock Information
                'stock_info': self._build_stock_info(),
                
                # Financial Data
                'financial_data': self._build_financial_data(),
                
                # Forensic Analysis
                'forensic_score': 0,
                'red_flags': [],
                
                # Executive Summary
                'executive_summary': {},
                
                # Valuation
                'valuation': {},
            }
            
            # Run forensic analysis
            score, red_flags, summary = run_forensic_analysis(self.ticker)
            self.report_data['forensic_score'] = score
            self.report_data['red_flags'] = red_flags
            self.report_data['forensic_summary'] = summary
            
            # Generate executive summary & recommendation
            self.report_data['executive_summary'] = self._build_executive_summary()
            
            # Build valuation
            self.report_data['valuation'] = self._build_valuation()
            
            logger.info(f"Research report generated for {self.ticker}")
            return self.report_data
            
        except Exception as e:
            logger.error(f"Failed to generate report for {self.ticker}: {e}")
            raise
    
    def _get_company_name(self) -> str:
        """Extract company name from info."""
        return self.info.get('longName') or self.info.get('shortName') or self.ticker
    
    def _build_stock_info(self) -> Dict:
        """Build stock information section."""
        current_price = self.info.get('currentPrice') or self.info.get('regularMarketPrice') or 0
        market_cap = self.info.get('marketCap', 0)
        
        # Determine market cap category
        market_cap_crore = market_cap / 10000000  # Convert to crores
        
        if market_cap_crore >= 50000:
            category = "Large Cap"
        elif market_cap_crore >= 10000:
            category = "Mid Cap"
        elif market_cap_crore >= 2500:
            category = "Small Cap"
        else:
            category = "Micro Cap"
        
        return {
            'name': self._get_company_name(),
            'ticker': self.ticker,
            'display_ticker': self.ticker,
            'exchange': self.info.get('exchange', 'NSE'),
            'sector': self.info.get('sector', 'Unknown'),
            'industry': self.info.get('industry', 'Unknown'),
            'current_price': current_price,
            'market_cap': market_cap,
            'market_cap_crore': market_cap_crore,
            'market_cap_category': category,
            'high_52_week': self.info.get('fiftyTwoWeekHigh', current_price),
            'low_52_week': self.info.get('fiftyTwoWeekLow', current_price),
            'avg_volume': self.info.get('averageVolume', 0),
            'beta': self.info.get('beta', 1.0),
        }
    
    def _build_financial_data(self) -> Dict:
        """Build financial data section with historical metrics."""
        try:
            # Fetch financial statements
            balance_sheet = self.stock.balance_sheet
            income_stmt = self.stock.income_stmt
            cash_flow = self.stock.cashflow
            
            # Extract years (columns)
            years = []
            if not income_stmt.empty:
                years = [col.strftime("%Y") for col in income_stmt.columns[:5]]
            
            # Revenue
            revenue = []
            if not income_stmt.empty and 'Total Revenue' in income_stmt.index:
                revenue = (income_stmt.loc['Total Revenue'] / 10000000).tolist()[:5]  # Convert to crores
            
            # EBITDA
            ebitda = []
            if not income_stmt.empty and 'EBITDA' in income_stmt.index:
                ebitda = (income_stmt.loc['EBITDA'] / 10000000).tolist()[:5]
            elif not income_stmt.empty:
                # Calculate EBITDA if not directly available
                try:
                    operating_income = income_stmt.loc['Operating Income'] if 'Operating Income' in income_stmt.index else 0
                    da = income_stmt.loc['Reconciled Depreciation'] if 'Reconciled Depreciation' in income_stmt.index else 0
                    ebitda = ((operating_income + da) / 10000000).tolist()[:5]
                except:
                    ebitda = [0] * len(years)
            
            # Net Income (PAT)
            pat = []
            if not income_stmt.empty and 'Net Income' in income_stmt.index:
                pat = (income_stmt.loc['Net Income'] / 10000000).tolist()[:5]
            
            # Operating Cash Flow
            cfo = []
            if not cash_flow.empty and 'Operating Cash Flow' in cash_flow.index:
                cfo = (cash_flow.loc['Operating Cash Flow'] / 10000000).tolist()[:5]
            
            # Debt & Equity
            total_debt = []
            equity = []
            if not balance_sheet.empty:
                if 'Total Debt' in balance_sheet.index:
                    total_debt = (balance_sheet.loc['Total Debt'] / 10000000).tolist()[:5]
                if 'Stockholders Equity' in balance_sheet.index:
                    equity = (balance_sheet.loc['Stockholders Equity'] / 10000000).tolist()[:5]
            
            # Shareholding pattern (if available)
            promoter_pct = [self.info.get('heldPercentInsiders', 0) * 100] * len(years)
            fii_pct = [self.info.get('heldPercentInstitutions', 0) * 100] * len(years)
            dii_pct = [0] * len(years)  # Not available in yfinance
            public_pct = [max(0, 100 - promoter_pct[0] - fii_pct[0])] * len(years)
            
            return {
                'years': years,
                'revenue': revenue,
                'ebitda': ebitda,
                'pat': pat,
                'cfo': cfo,
                'total_debt': total_debt,
                'equity': equity,
                'promoter_percent': promoter_pct,
                'fii_percent': fii_pct,
                'dii_percent': dii_pct,
                'public_percent': public_pct,
            }
            
        except Exception as e:
            logger.warning(f"Error building financial data: {e}")
            return {
                'years': [],
                'revenue': [],
                'ebitda': [],
                'pat': [],
                'cfo': [],
                'total_debt': [],
                'equity': [],
                'promoter_percent': [],
                'fii_percent': [],
                'dii_percent': [],
                'public_percent': [],
            }
    
    def _build_executive_summary(self) -> Dict:
        """
        Build executive summary with investment recommendation.
        Uses forensic score, financials, and valuation to determine rating.
        """
        forensic_score = self.report_data.get('forensic_score', 50)
        stock_info = self.report_data.get('stock_info', {})
        financial_data = self.report_data.get('financial_data', {})
        
        # Calculate growth rates
        revenue = financial_data.get('revenue', [])
        pat = financial_data.get('pat', [])
        
        revenue_cagr = self._calculate_cagr(revenue) if len(revenue) >= 2 else 0
        profit_cagr = self._calculate_cagr(pat) if len(pat) >= 2 else 0
        
        # Determine rating based on multiple factors
        rating = self._determine_rating(forensic_score, revenue_cagr, profit_cagr)
        
        # Calculate target price (simple valuation)
        current_price = stock_info.get('current_price', 0)
        target_price = self._calculate_target_price(current_price, rating, revenue_cagr)
        
        upside_potential = ((target_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Generate thesis & risks
        thesis = self._generate_investment_thesis(forensic_score, revenue_cagr, profit_cagr)
        catalysts = self._identify_catalysts()
        risks = self._identify_risks(forensic_score)
        
        return {
            'rating': rating,
            'target_price': target_price,
            'upside_potential': upside_potential,
            'investment_thesis': thesis,
            'key_catalysts': catalysts,
            'key_risks': risks,
            'recommendation': self._get_recommendation_text(rating, upside_potential),
        }
    
    def _calculate_cagr(self, values: List[float]) -> float:
        """Calculate CAGR from list of values."""
        if not values or len(values) < 2:
            return 0
        
        # Filter out zeros and negatives
        valid_values = [v for v in values if v > 0]
        if len(valid_values) < 2:
            return 0
        
        start_value = valid_values[-1]  # Oldest
        end_value = valid_values[0]     # Latest
        n_years = len(valid_values) - 1
        
        try:
            cagr = ((end_value / start_value) ** (1 / n_years) - 1) * 100
            return cagr
        except:
            return 0
    
    def _determine_rating(self, forensic_score: int, revenue_cagr: float, profit_cagr: float) -> str:
        """
        Determine investment rating based on multiple factors.
        Rating scale: STRONG BUY, BUY, HOLD, REDUCE, SELL
        """
        score = 0
        
        # Forensic score contribution (max 40 points)
        if forensic_score >= 80:
            score += 40
        elif forensic_score >= 60:
            score += 25
        elif forensic_score >= 40:
            score += 10
        else:
            score += 0
        
        # Revenue growth contribution (max 30 points)
        if revenue_cagr >= 20:
            score += 30
        elif revenue_cagr >= 15:
            score += 20
        elif revenue_cagr >= 10:
            score += 10
        elif revenue_cagr >= 5:
            score += 5
        
        # Profit growth contribution (max 30 points)
        if profit_cagr >= 20:
            score += 30
        elif profit_cagr >= 15:
            score += 20
        elif profit_cagr >= 10:
            score += 10
        elif profit_cagr >= 5:
            score += 5
        
        # Determine rating
        if score >= 80:
            return "STRONG BUY"
        elif score >= 60:
            return "BUY"
        elif score >= 40:
            return "HOLD"
        elif score >= 20:
            return "REDUCE"
        else:
            return "SELL"
    
    def _calculate_target_price(self, current_price: float, rating: str, revenue_cagr: float) -> float:
        """Calculate target price based on rating and growth."""
        multipliers = {
            "STRONG BUY": 1.25,
            "BUY": 1.15,
            "HOLD": 1.05,
            "REDUCE": 0.95,
            "SELL": 0.85,
        }
        
        base_multiplier = multipliers.get(rating, 1.0)
        
        # Adjust for growth
        growth_adjustment = min(revenue_cagr / 100, 0.15)  # Cap at 15% adjustment
        
        final_multiplier = base_multiplier + growth_adjustment
        
        return round(current_price * final_multiplier, 2)
    
    def _generate_investment_thesis(self, forensic_score: int, revenue_cagr: float, profit_cagr: float) -> str:
        """Generate AI-powered investment thesis."""
        company_name = self._get_company_name()
        sector = self.info.get('sector', 'the sector')
        
        if forensic_score >= 80 and revenue_cagr >= 15:
            return (f"{company_name} demonstrates strong fundamentals with a forensic score of {forensic_score}/100, "
                   f"indicating clean accounting and robust governance. The company has achieved {revenue_cagr:.1f}% "
                   f"revenue CAGR with {profit_cagr:.1f}% profit growth, positioning it as a quality compounder in {sector}. "
                   f"Strong cash conversion and healthy balance sheet provide downside protection.")
        
        elif forensic_score >= 60 and revenue_cagr >= 10:
            return (f"{company_name} presents a balanced investment case with acceptable fundamentals (forensic score: {forensic_score}/100). "
                   f"The company has delivered {revenue_cagr:.1f}% revenue growth, though certain financial metrics warrant monitoring. "
                   f"Suitable for investors seeking moderate growth with manageable risk in {sector}.")
        
        elif forensic_score >= 40:
            return (f"{company_name} shows mixed signals with a forensic score of {forensic_score}/100. "
                   f"While the business operates in {sector}, several red flags in accounting quality, leverage, or governance "
                   f"require thorough due diligence. Revenue growth of {revenue_cagr:.1f}% provides some support, but risks are elevated.")
        
        else:
            return (f"{company_name} exhibits significant quality concerns with a low forensic score of {forensic_score}/100. "
                   f"Multiple red flags in financial health, cash flow quality, or corporate governance suggest high risk. "
                   f"Not recommended for quality-focused portfolios. Only suitable for deep-value or turnaround specialists.")
    
    def _identify_catalysts(self) -> str:
        """Identify potential catalysts for stock."""
        sector = self.info.get('sector', '')
        industry = self.info.get('industry', '')
        
        # Generic catalysts - in production, would use AI to scan news/reports
        catalysts = [
            "Capacity expansion driving volume growth",
            "Margin recovery from operational leverage",
            "Market share gains in key segments",
            "New product launches or service offerings",
            "Favorable regulatory environment",
        ]
        
        return ", ".join(catalysts[:3])
    
    def _identify_risks(self, forensic_score: int) -> str:
        """Identify key investment risks."""
        risks = []
        
        if forensic_score < 70:
            risks.append("Financial statement quality concerns")
        
        if self.info.get('beta', 1.0) > 1.5:
            risks.append("High volatility (Beta > 1.5)")
        
        # Add generic risks
        risks.extend([
            "Competitive intensity and margin pressure",
            "Regulatory or policy changes",
            "Macroeconomic headwinds",
        ])
        
        return ", ".join(risks[:4])
    
    def _get_recommendation_text(self, rating: str, upside: float) -> str:
        """Get detailed recommendation text."""
        if rating in ["STRONG BUY", "BUY"]:
            return f"We recommend {rating} with {upside:.1f}% upside to target price. Suitable for long-term wealth creation."
        elif rating == "HOLD":
            return f"Maintain HOLD rating. Current valuation offers limited upside ({upside:.1f}%). Existing investors can hold."
        else:
            return f"We recommend {rating}. Risk-reward is unfavorable with {upside:.1f}% potential. Consider reducing exposure."
    
    def _build_valuation(self) -> Dict:
        """Build valuation section."""
        pe_ratio = self.info.get('trailingPE', 0)
        pb_ratio = self.info.get('priceToBook', 0)
        ps_ratio = self.info.get('priceToSalesTrailing12Months', 0)
        peg_ratio = self.info.get('pegRatio', 0)
        
        dividend_yield = self.info.get('dividendYield', 0) * 100 if self.info.get('dividendYield') else 0
        
        return {
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'ps_ratio': ps_ratio,
            'peg_ratio': peg_ratio,
            'dividend_yield': dividend_yield,
            'ev_to_ebitda': self.info.get('enterpriseToEbitda', 0),
        }


def generate_research_report(ticker: str, analyst_name: str = "Ace-Trade AI") -> Dict:
    """
    Main entry point for generating research reports.
    Returns complete report dictionary.
    """
    generator = ResearchReportGenerator(ticker)
    return generator.generate_report(analyst_name)
