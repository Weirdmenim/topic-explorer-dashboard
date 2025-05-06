# Topic Explorer Dashboard

An interactive NLP dashboard for exploring term frequencies and visualizing word clouds from any text corpus. Built with Streamlit, pandas, scikit-learn, and wordcloud, this tool lets users upload CSV/JSON datasets or ZIP archives of text files and instantly gain insights into the most frequent terms.

---

## Live Demo

🔗 [View the live dashboard on Streamlit Cloud](https://share.streamlit.io/yourusername/topic-explorer-dashboard/app.py)

---

## Screenshots

<details>
<summary>Click to expand</summary>

| ![Term Frequency Bar Chart](docs/bar_chart_example.png) | ![Word Cloud Visualization](docs/wordcloud_example.png) |
| :-----------------------------------------------------: | :-----------------------------------------------------: |
|    *Top 20 term frequencies for the uploaded corpus*    |           *Word cloud highlighting key terms*           |

</details>

---

## Features

* **Multiple Input Formats**: Upload CSV, JSON, or ZIP of `.txt` files.
* **Term-Frequency Analysis**: Compute top-N word counts with scikit-learn’s `CountVectorizer`.
* **Bar Chart**: Visualize term frequencies with an interactive bar chart.
* **Word Cloud**: Generate dynamic word clouds to highlight prominent terms.
* **Sidebar Controls**: Adjust Top-N terms interactively.
* **Source Table**: View list of source files when uploading a ZIP.

---

## Table of Contents

1. [Installation](#installation)
2. [Deployment](#deployment)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Testing](#testing)
6. [One-Slide Summary](#one-slide-summary)
7. [Future Enhancements](#future-enhancements)
8. [License](#license)

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/topic-explorer-dashboard.git
   cd topic-explorer-dashboard
   ```
2. **Create & activate virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .\.venv\Scripts\Activate   # Windows PowerShell
   ```
3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Run locally**:

   ```bash
   streamlit run app.py
   ```

---

## Deployment

To deploy on Streamlit Cloud:

1. Push your code to a public GitHub repository.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io).
3. Click **New app**, select your repo and branch, and set `app.py` as the main file.
4. Click **Deploy**—your dashboard will be live at `https://share.streamlit.io/<username>/topic-explorer-dashboard/app.py`.

---

## Usage

1. **Upload Corpus**: Click **Browse files** and select a CSV, JSON, or ZIP of `.txt` files.
2. **Adjust Top-N**: Use the sidebar slider to select how many top terms to display.
3. **View Charts**: The main panel shows a bar chart and word cloud for the selected terms.
4. **Sources Tab**: If using a ZIP, expand **Sources** at the bottom to see file names.

---

## Project Structure

```text
topic-explorer-dashboard/
├── app.py               # Streamlit entrypoint
├── explorer/
│   ├── __init__.py
│   ├── data.py          # Corpus loading & processing
│   └── viz.py           # Visualization functions
├── requirements.txt     # Python dependencies
├── tests/
│   ├── test_data.py
│   └── test_viz.py      # Unit tests for data & viz modules
├── docs/
│   ├── bar_chart_example.png
│   └── wordcloud_example.png
└── README.md            # This documentation
```

---

## Testing

Ensure all functionality remains robust:

```bash
pytest -q
```

You should see all tests pass for data ingestion, processing, and visualization.

---

## One-Slide Summary

**Topic Explorer Dashboard**
**Features:** Corpus upload | Term-frequency bar chart | Word cloud | Interactive controls
**Key Takeaway:** Rapidly built an end-to-end NLP dashboard using Streamlit and scikit-learn, covered by 100% test coverage for core modules.

---

## Future Enhancements

* **Metadata Filters:** Filter by date, category, or custom tags.
* **Parallel Processing:** Speed up ZIP ingestion with multiprocessing.
* **Export Options:** Download charts as images or data as CSV.
* **Advanced NLP:** Integrate topic modeling (LDA) or sentiment overlays.

---

## License

