"""
Knowledge Base Builder — Generates RAG documents from the dataset and
indexes all documents into ChromaDB for semantic retrieval.

Run this script once to build the knowledge base:
    python -m rag.build_knowledge_base
"""

import os
import pandas as pd
import datetime
from config.settings import DATA_PATH, CHROMA_DB_PATH, KNOWLEDGE_SOURCES_PATH, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_COLLECTION_NAME


def generate_market_insights(df: pd.DataFrame) -> str:
    """Auto-generate king_county_market_insights.md from actual dataset."""

    overall_median = df["price"].median()
    overall_mean = df["price"].mean()
    total_sales = len(df)

    lines = [
        "# King County Market Insights — Data-Driven Analysis\n",
        "## Overall Market Summary\n",
        f"Based on {total_sales:,} property sales recorded in King County, Washington (2014-2015):\n",
        f"- **Median Sale Price:** ${overall_median:,.0f}",
        f"- **Mean Sale Price:** ${overall_mean:,.0f}",
        f"- **Price Range:** ${df['price'].min():,.0f} to ${df['price'].max():,.0f}",
        f"- **Total Unique Zipcodes:** {df['zipcode'].nunique()}",
        f"- **Average Living Area:** {df['sqft_living'].mean():,.0f} sqft",
        f"- **Average Lot Size:** {df['sqft_lot'].mean():,.0f} sqft",
        f"- **Average Grade:** {df['grade'].mean():.1f}/13",
        f"- **Average Condition:** {df['condition'].mean():.1f}/5\n",
    ]

    # Price distribution by quartile
    q1, q2, q3 = df["price"].quantile([0.25, 0.5, 0.75])
    lines += [
        "## Price Distribution\n",
        f"- **25th Percentile:** ${q1:,.0f} (lower quartile boundary)",
        f"- **50th Percentile (Median):** ${q2:,.0f}",
        f"- **75th Percentile:** ${q3:,.0f} (upper quartile boundary)",
        f"- **Interquartile Range:** ${q3 - q1:,.0f}\n",
    ]

    # Top 10 most expensive zipcodes
    zip_stats = df.groupby("zipcode").agg(
        median_price=("price", "median"),
        mean_price=("price", "mean"),
        count=("price", "count"),
        avg_grade=("grade", "mean"),
    ).sort_values("median_price", ascending=False)

    lines += ["\n## Top 10 Most Expensive Zipcodes\n"]
    for i, (zc, row) in enumerate(zip_stats.head(10).iterrows()):
        lines.append(
            f"{i+1}. **Zipcode {int(zc)}:** Median ${row['median_price']:,.0f} | "
            f"Avg Grade {row['avg_grade']:.1f} | {int(row['count'])} sales"
        )

    # Top 10 most affordable
    lines += ["\n## Top 10 Most Affordable Zipcodes\n"]
    for i, (zc, row) in enumerate(zip_stats.tail(10).iterrows()):
        lines.append(
            f"{i+1}. **Zipcode {int(zc)}:** Median ${row['median_price']:,.0f} | "
            f"Avg Grade {row['avg_grade']:.1f} | {int(row['count'])} sales"
        )

    # Waterfront premium analysis
    wf = df[df["waterfront"] == 1]
    nwf = df[df["waterfront"] == 0]
    wf_premium = (wf["price"].median() / nwf["price"].median() - 1) * 100
    lines += [
        "\n## Waterfront Premium Analysis\n",
        f"- **Waterfront Properties:** {len(wf)} ({len(wf)/len(df)*100:.1f}% of all sales)",
        f"- **Waterfront Median Price:** ${wf['price'].median():,.0f}",
        f"- **Non-Waterfront Median Price:** ${nwf['price'].median():,.0f}",
        f"- **Waterfront Premium:** {wf_premium:.0f}% above non-waterfront median\n",
    ]

    # Grade impact analysis
    lines += ["\n## Price by Building Grade\n"]
    grade_stats = df.groupby("grade")["price"].agg(["median", "mean", "count"])
    for grade, row in grade_stats.iterrows():
        if row["count"] >= 10:
            lines.append(
                f"- **Grade {int(grade)}:** Median ${row['median']:,.0f} | "
                f"Mean ${row['mean']:,.0f} | {int(row['count'])} properties"
            )

    # View impact
    lines += ["\n## View Quality Impact on Price\n"]
    view_stats = df.groupby("view")["price"].agg(["median", "count"])
    for view, row in view_stats.iterrows():
        lines.append(
            f"- **View Rating {int(view)}:** Median ${row['median']:,.0f} | {int(row['count'])} properties"
        )

    # Condition impact
    lines += ["\n## Property Condition Impact on Price\n"]
    cond_stats = df.groupby("condition")["price"].agg(["median", "count"])
    for cond, row in cond_stats.iterrows():
        lines.append(
            f"- **Condition {int(cond)}:** Median ${row['median']:,.0f} | {int(row['count'])} properties"
        )

    return "\n".join(lines)


def generate_zipcode_profiles(df: pd.DataFrame) -> str:
    """Auto-generate zipcode_profiles.md with per-zipcode statistics."""

    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    df["house_age"] = datetime.datetime.now().year - df["yr_built"]
    df["renovated"] = df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)

    lines = [
        "# King County Zipcode Profiles\n",
        "Comprehensive statistics for each zipcode based on historical sales data.\n",
    ]

    zip_data = df.groupby("zipcode").agg(
        sales_count=("price", "count"),
        median_price=("price", "median"),
        mean_price=("price", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        price_std=("price", "std"),
        avg_sqft=("sqft_living", "mean"),
        avg_lot=("sqft_lot", "mean"),
        avg_grade=("grade", "mean"),
        avg_condition=("condition", "mean"),
        avg_ppsf=("price_per_sqft", "mean"),
        avg_bedrooms=("bedrooms", "mean"),
        avg_bathrooms=("bathrooms", "mean"),
        pct_waterfront=("waterfront", "mean"),
        avg_age=("house_age", "mean"),
        pct_renovated=("renovated", "mean"),
    ).sort_values("median_price", ascending=False)

    for zc, row in zip_data.iterrows():
        tier = (
            "Luxury" if row["median_price"] > 800000
            else "Premium" if row["median_price"] > 500000
            else "Mid-Market" if row["median_price"] > 300000
            else "Value"
        )
        lines += [
            f"\n## Zipcode {int(zc)} — {tier} Tier\n",
            f"- **Sales Volume:** {int(row['sales_count'])} properties",
            f"- **Median Price:** ${row['median_price']:,.0f}",
            f"- **Mean Price:** ${row['mean_price']:,.0f}",
            f"- **Price Range:** ${row['min_price']:,.0f} – ${row['max_price']:,.0f}",
            f"- **Price Volatility (Std Dev):** ${row['price_std']:,.0f}",
            f"- **Avg Living Area:** {row['avg_sqft']:,.0f} sqft",
            f"- **Avg Lot Size:** {row['avg_lot']:,.0f} sqft",
            f"- **Avg Grade:** {row['avg_grade']:.1f}/13",
            f"- **Avg Condition:** {row['avg_condition']:.1f}/5",
            f"- **Avg Price/Sqft:** ${row['avg_ppsf']:,.0f}",
            f"- **Avg Bedrooms:** {row['avg_bedrooms']:.1f}",
            f"- **Avg Bathrooms:** {row['avg_bathrooms']:.1f}",
            f"- **Waterfront Properties:** {row['pct_waterfront']*100:.1f}%",
            f"- **Avg House Age:** {row['avg_age']:.0f} years",
            f"- **Renovated Properties:** {row['pct_renovated']*100:.1f}%",
        ]

    return "\n".join(lines)


def chunk_document(text: str, source_name: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split a document into chunks, preferring markdown header boundaries."""

    sections = text.split("\n## ")
    chunks = []

    for i, section in enumerate(sections):
        if i > 0:
            section = "## " + section

        # Extract section header for metadata
        header = section.split("\n")[0].strip("# ").strip()

        # If section is short enough, keep as one chunk
        if len(section) <= chunk_size:
            chunks.append({
                "text": section.strip(),
                "metadata": {"source": source_name, "section": header},
            })
        else:
            # Split by paragraphs within section
            paragraphs = section.split("\n\n")
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {"source": source_name, "section": header},
                    })
                    # Keep overlap from end of previous chunk
                    current_chunk = current_chunk[-overlap:] + "\n\n" + para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {"source": source_name, "section": header},
                })

    return chunks


def build_kb():
    """Build the complete ChromaDB knowledge base from all sources."""
    import chromadb
    from rag.embeddings import get_embedding_function

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["bedrooms"] <= 10]

    # Step 1: Generate data-driven documents
    print("Generating market insights from dataset...")
    insights = generate_market_insights(df)
    insights_path = os.path.join(KNOWLEDGE_SOURCES_PATH, "king_county_market_insights.md")
    with open(insights_path, "w") as f:
        f.write(insights)
    print(f"  Written: {insights_path}")

    print("Generating zipcode profiles from dataset...")
    profiles = generate_zipcode_profiles(df)
    profiles_path = os.path.join(KNOWLEDGE_SOURCES_PATH, "zipcode_profiles.md")
    with open(profiles_path, "w") as f:
        f.write(profiles)
    print(f"  Written: {profiles_path}")

    # Step 2: Load all documents
    print("Loading all knowledge source documents...")
    all_chunks = []
    for filename in os.listdir(KNOWLEDGE_SOURCES_PATH):
        if filename.endswith(".md"):
            filepath = os.path.join(KNOWLEDGE_SOURCES_PATH, filename)
            with open(filepath, "r") as f:
                text = f.read()
            doc_name = filename.replace(".md", "").replace("_", " ").title()
            chunks = chunk_document(text, doc_name)
            all_chunks.extend(chunks)
            print(f"  {filename}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Step 3: Build ChromaDB
    print("Building ChromaDB collection...")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection if present
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    embedding_fn = get_embedding_function()
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # Add chunks in batches
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            ids=[f"chunk_{i + j}" for j, _ in enumerate(batch)],
        )

    print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' built with {len(all_chunks)} chunks")
    print(f"Stored at: {CHROMA_DB_PATH}")
    return collection


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_kb()
