import pickle

def inspect_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    # 打印顶层结构
    print("Top-level keys or types:", dir(data))  # 如果是字典或对象
    
    # 示例：查看顶点和边的部分数据
    if "vertices" in data:
        print("\nVertices (first 5):")
        print(data["vertices"][:5])
    
    if "edges" in data:
        print("\nFirst edge details:")
        first_edge = data["edges"][0]
        print("Type:", first_edge.get("type"))
        print("Is closed:", first_edge.get("is_closed"))
        print("Samples (first 3):", first_edge.get("samples")[:3])
        
# 使用示例
inspect_pickle("NerVE_dataset/00022787/step_edge_reso64.pkl")