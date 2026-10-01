# GenRec: LLM-Backed Recommendation Ranker

> **Based on the research paper:** "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
> by Ying Li, Shradha Sehgal, Arjun Rao, Rein Houthooft, Yunan Hu, Yaochen Zhu, Sourabh Medapati, Yun Li, Linas Baltrunas, Grace Huang, Ashish Rastogi, Kamelia Aryafar (Netflix, 2026)
> arXiv:2608.10257v2 [cs.IR]

---

## 🏗️ Project Builder

**Sandesh Dhital** — Creator and Builder of this implementation

## 📄 Original Paper Authors (Credit Where Due)

| Author | Affiliation |
|--------|-------------|
| Ying Li | Netflix |
| Shradha Sehgal | Netflix |
| Arjun Rao | Netflix |
| Rein Houthooft | Amazon (work done at Netflix) |
| Yunan Hu | Netflix |
| Yaochen Zhu | Netflix |
| Sourabh Medapati | Netflix |
| Yun Li | Netflix |
| Linas Baltrunas | Netflix |
| Grace Huang | Netflix |
| Ashish Rastogi | Netflix |
| Kamelia Aryafar | Netflix |

---

## 📖 About

This project implements the **GenRec** framework described in the paper — an LLM-backed recommendation ranker that replaces traditional feature-engineered ranking models with a foundation LLM approach.

### Key Concepts from the Paper

1. **Two-Phase Framework:**
   - **Phase 1:** Adapt an open-source LLM to domain data (content understanding + instruction following)
   - **Phase 2:** Post-train with recommendation-ranking specific data, labels, and reward signals

2. **Input Verbalization:** User histories and context are converted to natural language instead of engineered features

3. **Context Engineering:** Rich contextual signals (time, device, session behavior) are verbalized as text

4. **Reward Integration:** Business requirements and long-term member satisfaction are encoded as reward signals

5. **Cost-Constrained Serving:** Prefill-only inference approach for efficient serving

6. **Paradigm Shift:** From feature engineering → context engineering; from bespoke architectures → shared foundation backbones

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/iamsandeshdhital/GenRec.git
cd GenRec

# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py

# Start the web interface
python app.py
```

---

## 📁 Project Structure

```
GenRec/
├── README.md                  # This file
├── CREDITS.md                 # Detailed credits and attribution
├── requirements.txt           # Python dependencies
├── config.py                  # Configuration management
├── app.py                     # Web interface (Flask)
├── demo.py                    # Quick demo script
├── genrec/                    # Core library
│   ├── __init__.py
│   ├── verbalizer.py          # Input verbalization (user history → text)
│   ├── context_engine.py      # Context engineering module
│   ├── ranker.py              # LLM-backed ranking model
│   ├── reward.py              # Reward signal integration
│   ├── training.py            # Phase 1 & Phase 2 training pipeline
│   ├── serving.py             # Prefill-only inference serving
│   └── evaluation.py          # Offline & online evaluation metrics
├── data/                      # Sample data
│   ├── sample_catalog.json    # Sample content catalog
│   └── sample_interactions.json
├── templates/                 # Web UI templates
│   ├── index.html
│   └── results.html
├── static/                    # CSS/JS assets
│   └── style.css
└── tests/                     # Unit tests
    └── test_genrec.py
```

---

## 📊 Results (from Paper)

The paper reports that GenRec achieved:
- **Statistically significant gains** in offline and online metrics
- Trained with **substantially fewer** Phase-2 labeled examples
- **Reduced input signals** compared to traditional rankers
- Successful **large-scale A/B test** at Netflix

---

## 📜 License

This implementation was built by **Sandesh Dhital**. The original research is by Netflix.

---

## 🙏 Acknowledgments

- **Netflix Research** for publishing the GenRec paper
- The original authors for their groundbreaking work on LLM-backed recommenders
- Open-source LLM community for foundation models

---

## 📧 Contact

**Sandesh Dhital** — Project Builder
GitHub: [@iamsandeshdhital](https://github.com/iamsandeshdhital)
