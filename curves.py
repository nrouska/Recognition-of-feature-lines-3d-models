from vvrpywork.constants import *
from vvrpywork.scene import *
from vvrpywork.shapes import *
from random import random,seed
from scipy.spatial import ConvexHull
import numpy as np

import math

seed(42)
np.random.seed(42)
WIDTH = 800
HEIGHT = 800

class Project(Scene3D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Project")
        self.reset()

    def reset(self):
        self.model = Mesh3D("portrait2.ply", color = Color.GRAY)
        self.addShape(self.model, "model")

        self.adg_list_onehop = self.find_adjacency_list(self.model.triangles, len(self.model.vertices), hops=1)       
        self.adj_list = self.find_adjacency_list(self.model.triangles, len(self.model.vertices), hops=5)       
        self.Task1_classify_vertices()
        #self.Task2_3_colored_feature_curves()
        self.Task4_group_feature_curves()

      
    def Task1_classify_vertices(self):
        self.edges, self.corners, self.faces = self.patch_PCA(self.model.vertices, self.adj_list)
        print(f"Green Edges: {len(self.edges)}, Red Corners: {len(self.corners)}, Blue Faces: {len(self.faces)}")
        vc = self.model.vertex_colors
        vc[self.faces] = (0, 0, 1) #blue
        vc[self.edges] = (0, 1, 0) #green
        vc[self.corners] = (1, 0, 0) #red
        self.model.vertex_colors =vc
        self.updateShape("model")

    def Task2_3_colored_feature_curves(self):
        curves = self.extract_feature_curves(self.edges)
        vertices = self.model.vertices  
        print(f"Curves: {len(curves)}")
        for idx, curve in enumerate(curves):
            line_segments = []

            
            curve_points = np.array([vertices[v] for v in curve])
            
            for i in range(len(curve_points) - 1):
               line_segments.append([i, i + 1])
            # each curve has different color
            color = (random(), random(), random())

            lineset = LineSet3D(points=curve_points.tolist(), lines=line_segments, width=3, color=color)
            self.addShape(lineset, f"feature_curve_{idx}")


    def Task4_group_feature_curves(self):
        vertices = self.model.vertices
        curves = self.extract_feature_curves(self.edges)
      
        original_curves, groups = self.group_feature_curves(curves, vertices)
        print(f"Groups: {len(groups)}")
        print(groups)
        for group_idx, group in enumerate(groups):
            color = (random(), random(), random())  

            for curve_idx in group:
              
                curve = original_curves[curve_idx]
                ordered_curve = self.order_curve_points(curve, self.adg_list_onehop)
                curve_points = np.array([vertices[v] for v in ordered_curve])
                line_segments = [[i, i + 1] for i in range(len(curve_points) - 1)]

                lineset = LineSet3D(points=curve_points.tolist(), lines=line_segments, width=3, color=color)
                self.addShape(lineset, f"group_{group_idx}_curve_{curve_idx}")

                
    def extract_feature_curves(self, edge_indices):
        #O(1) lookup time in set
        edge_set = set(edge_indices)
        visited = set()
        curves = []

        for i in edge_indices:
            if i in visited:
                #already part of another curve
                continue

            curve = []
            #stack for depth first search
            stack = [i]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                curve.append(current)

                # curve is shaped by "edge" vertices that are directly connected
                for neighbor in  self.adg_list_onehop[current]:
                    if neighbor in edge_set and neighbor not in visited:
                        stack.append(neighbor)
            #only curves with more than 50 points
            if len(curve) > 50:
                curves.append(curve)

        # each curve is a list of connected edge indices
        return curves
    
    def order_curve_points(self, curve, onehop_adj):
        curve_set = set(curve)
        neighbors = {v: [n for n in onehop_adj[v] if n in curve_set] for v in curve}

        # Βρες ακραίο σημείο (1 γείτονα)
        end_points = [v for v in curve if len(neighbors[v]) == 1]
        if end_points:
            start = end_points[0]
        else:
            # αν είναι κύκλος, βρες το πιο "απομακρυσμένο" ζευγάρι
            max_dist = -1
            start = curve[0]
            for i in range(len(curve)):
                for j in range(i + 1, len(curve)):
                    d = np.linalg.norm(self.model.vertices[curve[i]] - self.model.vertices[curve[j]])
                    if d > max_dist:
                        max_dist = d
                        start = curve[i]

        ordered = [start]
        visited = set([start])
        current = start

        while True:
            unvisited_neighbors = [n for n in neighbors[current] if n not in visited]
            if not unvisited_neighbors:
                break
            # προτίμησε τον πιο ευθύ γείτονα (πιο ευθυγραμμισμένος με προηγούμενο βήμα)
            if len(ordered) >= 2:
                prev = ordered[-2]
                prev_vec = self.model.vertices[current] - self.model.vertices[prev]
                best_next = max(
                    unvisited_neighbors,
                    key=lambda n: np.dot(
                        prev_vec / np.linalg.norm(prev_vec),
                        (self.model.vertices[n] - self.model.vertices[current]) / np.linalg.norm(self.model.vertices[n] - self.model.vertices[current])
                    )
                )
            else:
                best_next = unvisited_neighbors[0]

            ordered.append(best_next)
            visited.add(best_next)
            current = best_next

        return ordered
    

    #principal component analysis: identify directions of maximum variance in a patch 
    def patch_PCA(self, vertices, adj_list):
        edges = []
        corners = []
        faces = []

        for i, neighbors in enumerate(adj_list):
            #skip if a vertex has lower than 10 neighbors
            if len(neighbors) <= 10:
                continue
            
            #array ((len(neighbors)),3)
            patch = vertices[neighbors]
            centroid = patch.mean(axis=0) #mean of each column, center of patch
            # in each patch vertices are sifted such that the center mass is (0,0,0)
            centered = patch - centroid

            #3x3 covariance matrix
            cov = np.cov(centered.T)
            eigvals = np.sort(np.linalg.eigvalsh(cov)) #l1<l2<l3
            total = eigvals.sum()
            if total < 1e-6:
                continue

            #normalizes eigenvalues to sum to 1, scale invariant
            l1, l2, l3 = eigvals / total

            #classify the vertex based on how the patch spreads in space
            
            if l1 / l3 > 0.08:
                '''the smallest eigenvalue is relatively large
                compared to the largest -> patch in 3D '''
                corners.append(i)
            elif l3 / (l1 + l2) > 4:
                #the largest eigenvalue is much bigger than the other two -> patch in 1D
                edges.append(i)
            elif l3 < 0.6:
                #the largest eigenvalue is small -> the whole patch low variance ->flat
                faces.append(i)

        #returns 3 arrays: indices of points classified as edges, corners, and faces
        return np.array(edges, dtype=int), np.array(corners, dtype=int), np.array(faces, dtype=int)
     
    
    def find_adjacency_list(self, triangles, num_vertices, hops): 
        #for each vertex, a set of direct neighbors
        first_hop = [set() for _ in range(num_vertices)]  
        for tri in triangles:
            i, j, k = tri
            first_hop[i].update([j, k]) #adds j, k vertices to the set of neighbors for vertex i
            first_hop[j].update([i, k])
            first_hop[k].update([i, j])

        #breadth first search
        adj_list = []

        for i in range(num_vertices):
            visited = set([i])
            frontier = set([i]) #frontier of vertices to explore start with i

            for j in range(hops):
                next_frontier = set() #the neighbors of the current frontier
                for v in frontier:
                    next_frontier.update(first_hop[v])
                next_frontier -= visited  #avoid revisiting
                visited.update(next_frontier)
                frontier = next_frontier

            visited.discard(i)  #remove self
            adj_list.append(np.array(list(visited), dtype=int))

        ## returns a list o arrays
        ## each array contains indices of vertices reachable in n hops from vertex i
        return adj_list
    
    def group_feature_curves(self, curves, vertices):
        features = [] # list of feature vectors
        original_curves = []

        for idx,curve in enumerate(curves):
            points = vertices[curve]
            if len(curve) < 25:
                continue
            # feature 1: total curve length
            length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            # feature 2: how bendy is the curve
            avg_curvature = self.compute_average_curvature(points)

            
            direction = self.compute_curve_direction_PCA(points)
            direction = direction / np.linalg.norm(direction)
            compactness = self.compute_compactness_2d(points)
            
            print(f"Curve {idx} direction: {direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f}")

            print(f"Curve {idx}: length={length:.2f}, curvature={avg_curvature:.3f}, compactness={compactness:.4f}")

            feature_vector = [length, avg_curvature, compactness, *direction]  
            features.append(feature_vector)
            original_curves.append(curve)

        features = np.array(features)

        # Διαχωρισμός σε 2 ομάδες χαρακτηριστικών
        group1 = features[:, :3]  # length, curvature, compactness
        group2 = features[:, 3:]  # direction vector (x, y, z)

        # Κανονικοποίηση ξεχωριστά
        group1 = (group1 - group1.mean(axis=0)) / (group1.std(axis=0) + 1e-8)
        group2 = (group2 - group2.mean(axis=0)) / (group2.std(axis=0) + 1e-8)

        # Συνένωση πίσω σε ένα array
        features = np.hstack([group1, group2])
        # curves with correlation ≥ 0.4 are grouped together
        groups = self.group_by_correlation(features, threshold=0.4)

        #groups is a list of lists of indices of curves that are highly correlated
        return original_curves, groups
    
    def pearson_correlation(self, vec1, vec2):
        return np.corrcoef(vec1, vec2)[0, 1]

    def group_by_correlation(self,features, threshold):
        n = len(features) #total feature vectors(number of curves)
        groups = []
        assigned = set() #index of curve already grouped

        for i in range(n):
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j in range(i + 1, n): #compare curve i with all the others
                if j in assigned:
                    continue
                # pearson correlation between feature vector i and j
                corr = self.pearson_correlation(features[i], features[j])
                if corr >= threshold:
                    group.append(j)
                    assigned.add(j)
            groups.append(group)
        return groups
    
    def compute_curve_direction_PCA(self,points):
        # Center τα σημεία
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigenvectors είναι στήλες, το πρώτο (μεγαλύτερο eigenvalue) είναι κύρια κατεύθυνση
        principal_direction = eigvecs[:, np.argmax(eigvals)]
        return principal_direction
    
    def compute_average_curvature(self, points):
        if len(points) < 3:
            return 0
        curvatures = []
        for i in range(1, len(points) - 1):
            p0, p1, p2 = points[i-1], points[i], points[i+1]
            v1 = p1 - p0
            v2 = p2 - p1
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0))
            curvature = angle / np.linalg.norm(v1)
            curvatures.append(curvature)
        return np.mean(curvatures)
    

    def compute_compactness_2d(self, points):
        if len(points) < 3:
            return 0
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered)
        projected = centered @ vh[:2].T

        try:
            hull = ConvexHull(projected)
            area = hull.area
            # υπολογισμός perimeter
            hull_points = projected[hull.vertices]
            diffs = np.diff(np.vstack([hull_points, hull_points[0]]), axis=0)
            perimeter = np.sum(np.linalg.norm(diffs, axis=1))

            if perimeter == 0:
                return 0
            return area / (perimeter ** 2)
        except:
            return 0
 

   
if __name__ == "__main__":
    app = Project()
    app.mainLoop()