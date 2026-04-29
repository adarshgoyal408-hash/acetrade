# 🚀 ACE-TRADE ENHANCED - DEPLOYMENT GUIDE

## Quick Start (5 Minutes)

### Step 1: Copy All Files
Copy these files to your deployment directory:
```
acetrade_enhanced/
├── app.py                    # Main application (ENHANCED)
├── database.py               # Research database layer
├── forensic_analysis.py      # Forensic engine  
├── research_report.py        # Report generator
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── DEPLOYMENT_GUIDE.md       # This file
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Add API Key for AI Features
Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

### Step 4: Run the App
```bash
streamlit run app.py
```

### Step 5: Access the App
Open browser to: `http://localhost:8501`

---

## 🎯 What's New

### Navigation
Look for **📊 Equity Research** in the sidebar menu (5th item from top)

### Key Features
1. **Research Report Generator** - Full institutional-grade reports
2. **Forensic Analysis** - 15+ red flag parameters
3. **AI Co-Pilot** - Claude 4 powered insights
4. **Report Library** - All reports saved to SQLite
5. **Database Stats** - Analytics on your research

---

## 📊 How to Generate Your First Report

1. Click **📊 Equity Research** in sidebar
2. Go to **"Generate Report"** tab
3. Enter a ticker: `RELIANCE.NS` or `TCS.NS`
4. Click **"Generate Research Report"**
5. Wait 30-60 seconds
6. Review the comprehensive report!

---

## 🔧 Troubleshooting

### Problem: "Equity Research modules not available"
**Solution**: Ensure all 4 files are in the same directory:
- app.py
- database.py  
- forensic_analysis.py
- research_report.py

### Problem: Import errors
**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Problem: Database errors
**Solution**: Delete and recreate database
```bash
rm acetrade_research.db
# Database will auto-recreate on next run
```

### Problem: No data for ticker
**Solution**: 
- Verify ticker format: `SYMBOL.NS` (NSE) or `SYMBOL.BO` (BSE)
- Try major stocks first: RELIANCE.NS, TCS.NS, INFY.NS
- Check if markets are open

---

## 🎨 Customization

### Change Analyst Name
In the "Generate Report" tab, edit the "Analyst Name" field

### Modify Forensic Thresholds
Edit `forensic_analysis.py` - each check has configurable thresholds

### Add More Parameters
Add new checks to `ForensicAnalyzer` class in `forensic_analysis.py`

---

## 💾 Database Location

All research reports are stored in:
```
./acetrade_research.db
```

Session data (portfolio, watchlist) stored in:
```
./acetrade_data.json
```

To backup your research:
```bash
# Backup database
cp acetrade_research.db acetrade_research_backup.db

# Backup session data
cp acetrade_data.json acetrade_data_backup.json
```

---

## 🔐 Security Notes

### API Keys
- Store Anthropic API key in `.streamlit/secrets.toml`
- Never commit secrets to Git
- Add to `.gitignore`:
```
.streamlit/secrets.toml
acetrade_research.db
acetrade_data.json
acetrade_errors.log
```

### Broker Integration
- All broker auth uses official OAuth flows
- Credentials encrypted and stored locally
- Never shared with third parties

---

## 📈 Performance Tips

### Faster Report Generation
- Markets open = faster data fetch
- Cache is enabled (5min TTL)
- Run during market hours for best speed

### Database Performance
- SQLite handles 1000s of reports easily
- Auto-indexes on ticker and date
- No maintenance required

---

## 🌐 Deployment Options

### Option 1: Local (Recommended for Personal Use)
```bash
streamlit run app.py
```
✅ Fastest, most secure
✅ Full control over data

### Option 2: Streamlit Cloud (Public Hosting)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add secrets in Cloud dashboard
4. Deploy!

⚠️ Database resets on each deploy

### Option 3: Docker (Production)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t acetrade .
docker run -p 8501:8501 acetrade
```

---

## 🆘 Support

### Check Logs
```bash
tail -f acetrade_errors.log
```

### Test Individual Modules
```bash
# Test forensic analysis
python -c "from forensic_analysis import run_forensic_analysis; print(run_forensic_analysis('RELIANCE.NS'))"

# Test database
python -c "from database import get_database_stats; print(get_database_stats())"

# Test report generation  
python -c "from research_report import generate_research_report; print(generate_research_report('TCS.NS'))"
```

### Common Issues
1. **yfinance timeout** - Try again, markets may be closed
2. **No financial data** - Some stocks lack complete data
3. **SQLite locked** - Close other instances accessing DB

---

## 📞 Getting Help

1. Check `README.md` for feature documentation
2. Review `acetrade_errors.log` for error details
3. Test components individually (see above)
4. Verify all files are present and up-to-date

---

## ✅ Verification Checklist

Before deployment, verify:

- [ ] All 4 Python files present
- [ ] requirements.txt installed
- [ ] App runs without errors
- [ ] Can navigate to Equity Research page
- [ ] Can generate a test report (RELIANCE.NS)
- [ ] Reports appear in Library tab
- [ ] Database stats showing data

---

## 🎉 You're All Set!

Your enhanced Ace-Trade platform is ready. You now have:

✅ Complete trading platform (all original features)  
✅ Professional equity research module  
✅ 15+ forensic analysis parameters  
✅ AI-powered insights  
✅ Persistent database storage  
✅ Report library and analytics  

**Happy Trading & Researching! 📊🚀**
