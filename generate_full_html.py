import os

md_path = r"C:\Users\Virendra\.gemini\antigravity\brain\5159bcc1-eed6-43f0-affa-8430b6048f05\FasalAstra_Complete_Report.md"
html_path = r"d:\THE DEVILS\FasalAstra_AI\Full_FasalAstra_Report.html"

with open(md_path, 'r', encoding='utf-8') as f:
    md_content = f.read()

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FasalAstra_AI Full Technical Report</title>
    <!-- Marked.js for Markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2e7d32;
            --primary-light: #e8f5e9;
            --secondary: #1b5e20;
            --text: #1a1a1a;
            --text-muted: #555;
            --bg: #ffffff;
            --border: #e0e0e0;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        body {
            font-family: 'Inter', sans-serif;
            line-height: 1.65;
            color: var(--text);
            margin: 0;
            background-color: #f1f5f1;
            padding: 40px 20px;
        }

        .report-wrapper {
            max-width: 1000px;
            margin: 0 auto;
            background: var(--bg);
            border-radius: 12px;
            box-shadow: var(--shadow);
            padding: 40px 60px;
        }

        /* Markdown Styles */
        h1, h2, h3, h4 { color: var(--secondary); margin-top: 1.5em; }
        h1 { border-bottom: 3px solid var(--primary); padding-bottom: 10px; font-size: 2.5em; color: var(--primary); }
        h2 { border-bottom: 2px solid var(--primary-light); padding-bottom: 8px; font-size: 1.8em; }
        p { margin-bottom: 1.2em; }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.95rem;
        }
        th {
            background-color: var(--primary-light);
            color: var(--secondary);
            font-weight: 600;
            text-align: left;
            padding: 14px;
            border: 1px solid var(--border);
        }
        td {
            padding: 12px 14px;
            border: 1px solid var(--border);
        }
        tr:nth-child(even) { background-color: #fafafa; }
        
        pre { background: #2d2d2d; color: #ccc; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; font-family: monospace; color: #333; }
        pre code { background: transparent; padding: 0; color: inherit; }
        
        blockquote { border-left: 5px solid var(--primary); margin: 20px 0; padding: 15px 20px; background: #e8f5e9; border-radius: 0 8px 8px 0; font-style: italic;}
        
        .floating-nav {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }
        .btn {
            background: var(--primary);
            color: white;
            padding: 12px 24px;
            border-radius: 30px;
            text-decoration: none;
            font-weight: 600;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Mermaid */
        .mermaid { background: white; margin: 30px 0; border: 1px solid var(--border); border-radius: 12px; padding: 25px; text-align: center; }

        @media print {
            .floating-nav { display: none !important; }
            body { background-color: white; padding: 0; }
            .report-wrapper { box-shadow: none; padding: 20px; }
            table, .mermaid { page-break-inside: avoid; }
            pre { page-break-inside: avoid; }
        }
    </style>
</head>
<body>

    <div class="floating-nav no-print">
        <button onclick="window.print()" class="btn">🖨️ Save as PDF</button>
    </div>

    <div class="report-wrapper" id="content">
        Loading report...
    </div>

    <!-- Hidden Raw Markdown Data -->
    <textarea id="raw-md" style="display:none;">"""

html_footer = """</textarea>

    <script>
        const rawMd = document.getElementById('raw-md').value;
        
        // Render Markdown to HTML
        document.getElementById('content').innerHTML = marked.parse(rawMd);
        
        // Convert pre > code.language-mermaid and code.language-xychart-beta into div.mermaid
        document.querySelectorAll('code.language-mermaid, code.language-xychart-beta').forEach(el => {
            const pre = el.parentElement;
            if(pre && pre.tagName.toLowerCase() === 'pre') {
                const div = document.createElement('div');
                div.className = 'mermaid';
                div.textContent = el.textContent;
                pre.parentElement.replaceChild(div, pre);
            }
        });

        // Initialize Mermaid
        mermaid.initialize({ startOnLoad: true, theme: 'default' });
        setTimeout(() => { mermaid.contentLoaded(); }, 500);
    </script>
</body>
</html>
"""

# Write to Full_FasalAstra_Report.html
with open(html_path, 'w', encoding='utf-8') as f:
    # Need to replace </textarea> inside markdown body if it exists, to stop early termination of the tag
    safe_md = md_content.replace("</textarea>", "&lt;/textarea&gt;")
    f.write(html_template + "\n" + safe_md + "\n" + html_footer)
print(f"Report generated successfully at {html_path}")
