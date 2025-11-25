"""
Helper script for querying the robotics papers database
Just some useful queries I keep running while working on the project
"""

import sqlite3
import sys
from pathlib import Path


def connect_db(db_path="papers.db"):
    """Connect to the database"""
    if not Path(db_path).exists():
        print(f"ERROR: Database not found at {db_path}")
        print("Run populate_database.py first!")
        sys.exit(1)
    return sqlite3.connect(db_path)


def most_popular_datasets(conn, limit=10):
    """Which datasets are used the most?"""
    print(f"\n{'='*60}")
    print(f"📊 Most Popular Datasets (Top {limit})")
    print(f"{'='*60}\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d.name, COUNT(DISTINCT pd.paper_id) as papers_using_it
        FROM datasets d
        JOIN paper_datasets pd ON d.id = pd.dataset_id
        GROUP BY d.name
        ORDER BY papers_using_it DESC
        LIMIT ?
    """, (limit,))
    
    results = cursor.fetchall()
    for i, (dataset, count) in enumerate(results, 1):
        print(f"{i:2d}. {dataset:30s} - used in {count} papers")


def robot_platforms(conn):
    """What robots are people using?"""
    print(f"\n{'='*60}")
    print("🤖 Robot Platforms Used")
    print(f"{'='*60}\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.name, COUNT(DISTINCT pr.paper_id) as num_papers
        FROM robots r
        JOIN paper_robots pr ON r.id = pr.robot_id
        GROUP BY r.name
        ORDER BY num_papers DESC
    """)
    
    results = cursor.fetchall()
    for robot, count in results:
        print(f"  • {robot:30s} ({count} papers)")


def papers_by_year(conn):
    """Papers per year breakdown"""
    print(f"\n{'='*60}")
    print("📅 Papers by Year")
    print(f"{'='*60}\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT year, COUNT(*) as count
        FROM papers
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year DESC
    """)
    
    results = cursor.fetchall()
    for year, count in results:
        print(f"  {year}: {'█' * count} ({count})")


def models_with_sizes(conn):
    """Show papers that report model sizes"""
    print(f"\n{'='*60}")
    print("🧠 Models with Reported Sizes")
    print(f"{'='*60}\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT paper_id, model_name, model_size, model_architecture
        FROM papers
        WHERE model_size IS NOT NULL
    """)
    
    results = cursor.fetchall()
    if results:
        for paper_id, model_name, model_size, arch in results:
            arch_str = f" ({arch})" if arch else ""
            print(f"  • {paper_id:20s} - {model_name}: {model_size}{arch_str}")
    else:
        print("  No papers with model sizes found :(")


def compute_hardware_used(conn):
    """What GPUs/TPUs are used?"""
    print(f"\n{'='*60}")
    print("💻 Compute Hardware")
    print(f"{'='*60}\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT h.name, COUNT(DISTINCT ph.paper_id) as num_papers
        FROM hardware h
        JOIN paper_hardware ph ON h.id = ph.hardware_id
        WHERE h.type = 'compute'
        GROUP BY h.name
        ORDER BY num_papers DESC
    """)
    
    results = cursor.fetchall()
    if results:
        for hw, count in results:
            print(f"  • {hw:15s} - {count} papers")
    else:
        print("  No compute hardware info found")


def baseline_comparisons(conn):
    """Which models are commonly used as baselines?"""
    print(f"\n{'='*60}")
    print("📈 Most Common Baseline Models")
    print(f"{'='*60}\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT b.name, COUNT(*) as times_compared
        FROM baselines b
        JOIN paper_baselines pb ON b.id = pb.baseline_id
        GROUP BY b.name
        ORDER BY times_compared DESC
    """)
    
    results = cursor.fetchall()
    for baseline, count in results:
        print(f"  • {baseline:20s} - compared {count} times")


def search_papers(conn, keyword):
    """Search for papers by keyword in title"""
    print(f"\n{'='*60}")
    print(f"🔍 Papers matching '{keyword}'")
    print(f"{'='*60}\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT paper_id, title, year, model_name
        FROM papers
        WHERE title LIKE ? OR paper_id LIKE ?
    """, (f'%{keyword}%', f'%{keyword}%'))
    
    results = cursor.fetchall()
    if results:
        for paper_id, title, year, model in results:
            year_str = f"({year})" if year else "(N/A)"
            model_str = f" - Model: {model}" if model else ""
            print(f"  • {paper_id}")
            print(f"    {title[:70]}... {year_str}{model_str}\n")
    else:
        print(f"  No papers found matching '{keyword}'")


def get_paper_details(conn, paper_id):
    """Get all details for a specific paper"""
    print(f"\n{'='*60}")
    print(f"📄 Paper Details: {paper_id}")
    print(f"{'='*60}\n")
    
    cursor = conn.cursor()
    
    # Basic info
    cursor.execute("""
        SELECT title, year, venue, model_name, model_size,
               model_architecture, vision_encoder, base_model,
               optimizer, learning_rate, batch_size, epochs,
               training_data_size, success_rate
        FROM papers
        WHERE paper_id = ?
    """, (paper_id,))
    
    result = cursor.fetchone()
    if not result:
        print(f"  Paper '{paper_id}' not found!")
        return
    
    (title, year, venue, model_name, model_size, arch, 
     vision_enc, base_model, opt, lr, bs, epochs, data_size, success) = result
    
    print(f"Title: {title}")
    if year:
        print(f"Year: {year}")
    if venue:
        print(f"Venue: {venue}")
    
    print("\n--- Model Info ---")
    if model_name:
        print(f"Model: {model_name} ({model_size or 'size not reported'})")
    if arch:
        print(f"Architecture: {arch}")
    if vision_enc:
        print(f"Vision Encoder: {vision_enc}")
    if base_model:
        print(f"Base Model: {base_model}")
    
    if opt or lr or bs or epochs:
        print("\n--- Training Details ---")
        if opt:
            print(f"Optimizer: {opt}")
        if lr:
            print(f"Learning Rate: {lr}")
        if bs:
            print(f"Batch Size: {bs}")
        if epochs:
            print(f"Epochs: {epochs}")
    
    if data_size:
        print(f"\nTraining Data Size: {data_size}")
    if success:
        print(f"Success Rate: {success}")
    
    # Datasets
    cursor.execute("""
        SELECT DISTINCT d.name, pd.dataset_type
        FROM datasets d
        JOIN paper_datasets pd ON d.id = pd.dataset_id
        WHERE pd.paper_id = ?
    """, (paper_id,))
    datasets = cursor.fetchall()
    if datasets:
        print("\n--- Datasets ---")
        for ds, ds_type in datasets:
            print(f"  • {ds} ({ds_type})")
    
    # Robots
    cursor.execute("""
        SELECT r.name
        FROM robots r
        JOIN paper_robots pr ON r.id = pr.robot_id
        WHERE pr.paper_id = ?
    """, (paper_id,))
    robots = cursor.fetchall()
    if robots:
        print("\n--- Robot Platforms ---")
        for (robot,) in robots:
            print(f"  • {robot}")
    
    # Baselines
    cursor.execute("""
        SELECT b.name
        FROM baselines b
        JOIN paper_baselines pb ON b.id = pb.baseline_id
        WHERE pb.paper_id = ?
    """, (paper_id,))
    baselines = cursor.fetchall()
    if baselines:
        print("\n--- Compared Against ---")
        for (baseline,) in baselines:
            print(f"  • {baseline}")


def main():
    db_path = "papers.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    conn = connect_db(db_path)
    
    print("\n" + "="*60)
    print("🤓 Robotics Papers Database Query Tool")
    print("="*60)
    
    # Run all the summary queries
    most_popular_datasets(conn, limit=10)
    robot_platforms(conn)
    papers_by_year(conn)
    models_with_sizes(conn)
    compute_hardware_used(conn)
    baseline_comparisons(conn)
    
    # Example paper detail
    print("\n" + "="*60)
    print("📖 Example: Detailed view of OpenVLA paper")
    get_paper_details(conn, "openvla")
    
    conn.close()
    
    print("\n" + "="*60)
    print("\n💡 To query specific papers, use:")
    print("   python query_database.py")
    print("\nOr use sqlite3 directly:")
    print(f"   sqlite3 {db_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

