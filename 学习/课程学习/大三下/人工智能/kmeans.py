import os

import numpy as np

LOCAL_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", LOCAL_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", LOCAL_CACHE_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def k_means_custom(data, k, max_iters=100, tol=1e-4, random_state=42):
    """
    使用 NumPy 实现 K-means 聚类。

    参数:
        data: shape = (n_samples, n_features) 的数据矩阵
        k: 聚类数量
        max_iters: 最大迭代次数
        tol: 质心移动阈值，小于该值则停止
        random_state: 随机种子，保证结果可复现
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("data 必须是二维数组")
    if k <= 0 or k > len(data):
        raise ValueError("k 必须满足 1 <= k <= 样本数")

    rng = np.random.default_rng(random_state)
    centroids = data[rng.choice(data.shape[0], size=k, replace=False)].copy()

    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) == 0:
                # 空簇时重新选择离现有质心最远的点，避免 NaN。
                min_distances = np.min(distances, axis=1)
                farthest_point_idx = np.argmax(min_distances)
                new_centroids[i] = data[farthest_point_idx]
            else:
                new_centroids[i] = cluster_points.mean(axis=0)

        centroid_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if centroid_shift < tol:
            break

    final_distances = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    labels = np.argmin(final_distances, axis=1)
    inertia = np.sum((data - centroids[labels]) ** 2)
    return centroids, labels, inertia


def plot_clusters(data, labels, centroids, output_path="kmeans_result.png"):
    """绘制聚类结果并保存图片。"""
    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for color, label in zip(colors, unique_labels):
        cluster_points = data[labels == label]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=100,
            color=color,
            label=f"Cluster {label}",
            edgecolors="black",
            alpha=0.85,
        )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=260,
        c="red",
        marker="X",
        label="Centroids",
        edgecolors="black",
    )

    for idx, point in enumerate(data):
        plt.annotate(
            f"P{idx}",
            (point[0], point[1]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    plt.title("K-means Clustering Result")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    # 模拟小镇居民数据 (x, y 坐标)
    residents = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0],
    ])

    centroids, labels, inertia = k_means_custom(residents, k=3, random_state=42)

    print("最终车辆停放地点:")
    print(centroids)
    print("\n每位居民所属簇:")
    print(labels)
    print(f"\n簇内平方和 (inertia): {inertia:.4f}")

    output_path = "kmeans_result.png"
    plot_clusters(residents, labels, centroids, output_path=output_path)
    print(f"\n可视化结果已保存到: {output_path}")
