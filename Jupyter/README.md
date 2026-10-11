# 📓 Jupyter Notebook Course

A comprehensive 5-chapter course covering Jupyter Notebooks — from fundamentals to best practices.

---

## 📋 Course Syllabus

### Chapter 1: Jupyter Fundamentals
> **File:** `01_jupyter_fundamentals.ipynb`

- What is Jupyter — history from IPython to Jupyter
- Notebook architecture (kernel, cells, JSON format)
- Keyboard shortcuts cheat sheet (Esc/Enter modes, A/B, DD, Shift+Enter)
- Cell types: Code, Markdown, Raw
- Magic commands: `%timeit`, `%%time`, `%who`, `%whos`, `%reset`, `%run`, `%load`
- Shell commands with `!`

### Chapter 2: Markdown & Rich Output in Jupyter
> **File:** `02_markdown_rich_output.ipynb`

- Markdown syntax (headers, lists, bold, italic, links, tables)
- LaTeX math equations (`$inline$` and `$$block$$`)
- Displaying images and HTML
- `IPython.display`: `display()`, `HTML()`, `Markdown()`, `Image()`
- Rich output: `_repr_html_()`, `_repr_latex_()`

### Chapter 3: Jupyter Productivity & Magic Commands
> **File:** `03_productivity_magic_commands.ipynb`

- Line magics vs cell magics
- `%timeit` and `%%timeit` for benchmarking
- `%matplotlib inline`
- `%store` for persisting variables between notebooks
- `%%writefile` to write cell contents to a file
- `%%capture` to suppress output
- `%env` for environment variables
- `%debug` and `%pdb` for interactive debugging
- `!pip install` from within notebooks

### Chapter 4: Data Analysis Workflow in Jupyter
> **File:** `04_data_analysis_workflow.ipynb`

- Complete data analysis using **only the standard library**
- Create a synthetic sales dataset (CSV) with 500 transactions
- Read CSV with the `csv` module
- Compute statistics (mean, median, mode, std dev) with `statistics`
- Group and aggregate data with `collections.defaultdict`
- Frequency analysis with `collections.Counter`
- Top products by revenue, monthly trends, category & regional breakdowns

### Chapter 5: Best Practices & Tips
> **File:** `05_best_practices_tips.ipynb`

- Notebook organization: title, imports, sections
- Version control: `.gitignore`, `nbstripout`, `jupytext`
- Common pitfalls: out-of-order execution, hidden state, kernel restart
- Exporting: `nbconvert` to HTML, PDF, Python script
- Jupyter extensions overview (classic & JupyterLab)
- When NOT to use notebooks (production code, large projects)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Installation

```bash
pip install jupyter
```

### Running the Notebooks

```bash
# Start Jupyter
cd python_course/Jupyter/
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Regenerating the Course

```bash
python generate_jupyter_course.py
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `generate_jupyter_course.py` | Script that generates all notebooks |
| `01_jupyter_fundamentals.ipynb` | Chapter 1 notebook |
| `02_markdown_rich_output.ipynb` | Chapter 2 notebook |
| `03_productivity_magic_commands.ipynb` | Chapter 3 notebook |
| `04_data_analysis_workflow.ipynb` | Chapter 4 notebook |
| `05_best_practices_tips.ipynb` | Chapter 5 notebook |
| `README.md` | This file |

---

## 📖 How to Use This Course

1. **Read the markdown cells** — they explain concepts with examples
2. **Run the code cells** — press `Shift+Enter` to execute
3. **Experiment** — modify code and re-run to explore
4. **Chapter 4** has a hands-on project — work through it step by step
5. **Chapter 5** has tips you'll reference again and again

> 💡 **Tip:** Use `Kernel → Restart & Run All` to verify notebooks run correctly from top to bottom.
