"""Debug retrieval failure for reg_a_014."""
import os, sys, json, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import PROCESSED_DIR
from agent.preprocessor import resolve_doc_path, preprocess_document
from agent.indexer import build_keyword_index
from agent.retriever import extract_keywords_from_question, stage1_retrieve

# Load reg_a_014
with open("public_dataset_upload/questions/group_a/regulatory_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)
q = next(qq for qq in questions if qq["qid"] == "reg_a_014")

question = q["question"]
doc_ids = q["doc_ids"]
print(f"Question: {question}")
print()

# Extract keywords
keywords = extract_keywords_from_question(question)
print(f"Keywords extracted ({len(keywords)}):")
for kw in keywords:
    print(f"  '{kw}'")
print()

# Build index
keyword_index = {}
for doc_id in doc_ids:
    md_path = os.path.join(PROCESSED_DIR, f"{doc_id}.md")
    if not os.path.exists(md_path):
        path = resolve_doc_path("regulatory", doc_id)
        md_path = preprocess_document(path, os.path.splitext(path)[1])
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    keyword_index.update(build_keyword_index(doc_id, text))

# Show doc content samples
for doc_id in doc_ids:
    entries = keyword_index.get(doc_id, [])
    print(f"Doc {doc_id}: {len(entries)} paragraphs")
    if entries:
        print(f"  First 2 paragraphs:")
        for e in entries[:2]:
            print(f"    [{e['para_id']}] {e['text'][:150]}...")
    print()

# Check if any keyword matches
print("Keyword match check:")
all_text = ""
for doc_id in doc_ids:
    for e in keyword_index.get(doc_id, []):
        all_text += e["text"]

for kw in keywords:
    if kw in all_text:
        pos = all_text.find(kw)
        print(f"  '{kw}' => FOUND at pos {pos}")
    else:
        print(f"  '{kw}' => NOT FOUND")

# Try broader searches
print("\nBroader searches:")
for term in ["股东", "大会", "董事会", "担保", "募集", "章程", "职权"]:
    results = stage1_retrieve(keyword_index, doc_ids, term)
    print(f"  '{term}': {len(results)} hits")
