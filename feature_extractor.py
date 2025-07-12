from vvrpywork.constants import *
from vvrpywork.scene import *
from vvrpywork.shapes import *
from random import random,seed
from scipy.spatial import ConvexHull
import numpy as np


seed(42)
np.random.seed(42)
WIDTH = 800
HEIGHT = 800

class FeatureCurves(Scene3D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Project")

    def load_model(self, path):
        self.model = Mesh3D(path, color=Color.GRAY)
        self.addShape(self.model, "model")
        return self.model


    def reset(self):
        self.adg_list_onehop = self.find_adjacency_list(self.model.triangles, len(self.model.vertices), hops=1)       
        self.adj_list = self.find_adjacency_list(self.model.triangles, len(self.model.vertices), hops=5)       
        self.Task1_classify_vertices()
 
        self.curves = self.extract_feature_curves(self.edges,self.model.vertices)
        #self.Task2_3_colored_feature_curves()
        original_curves, groups, features = self.group_feature_curves(self.curves, self.model.vertices)
        self.features = features  # αποθήκευση σε attribute
        self.groups = groups
        self.original_curves = original_curves
       
        self.Task4_group_feature_curves()

      
    def Task1_classify_vertices(self):
        self.edges, self.corners, self.faces = self.patch_PCA(self.model.vertices, self.adj_list)
        print(f"Points in edges: {len(self.edges)},Points in Corners: {len(self.corners)}, Points in Faces: {len(self.faces)}")
        vc = self.model.vertex_colors
        vc[self.faces] = (0, 0, 1) #blue
        vc[self.edges] = (0, 1, 0) #green
        vc[self.corners] = (1, 0, 0) #red
        self.model.vertex_colors =vc
        self.updateShape("model")

    def Task2_3_colored_feature_curves(self):
        
        vertices = self.model.vertices  
        print(f"Curves: {len(self.curves)}")
        for idx, curve in enumerate(self.curves):
            line_segments = []

            ordered_curve = self.order_curve_points(curve, self.adg_list_onehop)
            curve_points = np.array([vertices[v] for v in ordered_curve])
            
            max_dist = 0.05 

            for i in range(len(curve_points) - 1):
                p1 = curve_points[i]
                p2 = curve_points[i + 1]
                dist = np.linalg.norm(p1 - p2)

                if dist < max_dist:
                    line_segments.append([i, i + 1])

            # each curve has different color
            color = (random(), random(), random())

            lineset = LineSet3D(points=curve_points.tolist(), lines=line_segments, width=3, color=color)
            self.addShape(lineset, f"feature_curve_{idx}")


    def Task4_group_feature_curves(self):
        vertices = self.model.vertices
        print(f"Objects: {len(self.groups)}")
        
        for group_idx, group in enumerate(self.groups):
            color = (random(), random(), random())  

            for curve_idx in group:
                line_segments =[]
                curve = self.original_curves[curve_idx]
                ordered_curve = self.order_curve_points(curve, self.adg_list_onehop)
                curve_points = np.array([vertices[v] for v in ordered_curve])
                max_dist = 0.05 

                for i in range(len(curve_points) - 1):
                    p1 = curve_points[i]
                    p2 = curve_points[i + 1]
                    dist = np.linalg.norm(p1 - p2)

                    if dist < max_dist:
                        line_segments.append([i, i + 1])

                lineset = LineSet3D(points=curve_points.tolist(), lines=line_segments, width=3, color=color)
                name = f"group_{group_idx}_curve_{curve_idx}"
                self.addShape(lineset, name)

                
    def extract_feature_curves(self, edge_indices, vertices):
         #O(1) lookup time in set
        edge_set = set(edge_indices)
        visited = set()
        curves = []
        cos_threshold = np.cos(np.radians(50))

        for i in edge_indices:
            #already part of another curve
            if i in visited:
                continue
            #initialize a curve with i
            curve = [i]
            visited.add(i)

            # stack holds (current_point, previous_point) for direction checking
            stack = [(i, None)]

            while stack:
                current, prev = stack.pop()

                for neighbor in self.adg_list_onehop[current]:
                    if neighbor in edge_set and neighbor not in visited:
    
                        if prev is not None:
                            #direction vectors , current-prev, current-neighbor
                            v1 = vertices[current] - vertices[prev]
                            v2 = vertices[neighbor] - vertices[current]

                            # normalize (unit length)
                            v1 /= np.linalg.norm(v1) + 1e-8
                            v2 /= np.linalg.norm(v2) + 1e-8
                            #cos of the angle between the vectors
                            cos_angle = np.dot(v1, v2)
                            # if the angle is smaller than the threshold -uneven curve
                            if cos_angle < cos_threshold:
                                continue  

                        visited.add(neighbor)
                        curve.append(neighbor)
                        stack.append((neighbor, current))
            #keep the curve if its long enough
            if len(curve) > 25:
                curves.append(curve)

        # list of lists with the edge-points indices that form each curve
        return curves
    

    def order_curve_points(self, curve, onehop_adj):
        curve_set = set(curve)
        #for each point keep neighbors that belong to the curve
        neighbors = {v: [n for n in onehop_adj[v] if n in curve_set] for v in curve}

        #only 1 neighbor start or end of the curve
        end_points = [v for v in curve if len(neighbors[v]) == 1]
        if not end_points: #closed curve
            start = curve[0]  
        else:
            start = end_points[0]

        ordered = []
        visited = set()
        stack = [start]

        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            ordered.append(v)
            for u in neighbors[v]:
                if u not in visited:
                    stack.append(u)

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
        symmetry_axis = np.mean(vertices[:, 0])
        for curve in curves:
            points = vertices[curve]
            # feature 1: total curve length
            length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            # feature 2: how bendy is the curve
            avg_curvature = self.compute_average_curvature(points)
            # feature 3: principal axis of the curve (direction of maximum variance)
            direction = self.compute_curve_direction_PCA(points)
            #normalize i need only direction, scale invariant
            direction = direction / np.linalg.norm(direction)
           
            #how compact is the curve
            compactness = self.compute_compactness_2d(points)
        

            feature_vector = [ length, avg_curvature, compactness,*direction]  
            features.append(feature_vector)
            original_curves.append(curve)

        features = np.array(features)
        if features.ndim == 1:
            features = np.stack(features, axis=0)
        #normalization features in a vector mast be in the same scale for justice
        ''' η σύγκριση εξαρτάται από τη σχετική διαφορά των χαρακτηριστικών μεταξύ τους
         όχι από το ποιο έχει τις μεγαλύτερες αριθμητικές τιμές'''
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        #curves that have similarity>=0.5 add in the same group
        groups = self.group_by_correlation(features, threshold=0.5)

        #groups is a list of lists of indices of curves that are highly correlated
        return original_curves, groups, features
    
    #returns cosine of the angle between the two vectors
    def cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)

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
                corr = self.cosine_similarity(features[i], features[j])
                if corr >= threshold:
                    group.append(j)
                    assigned.add(j)
            groups.append(group)
        return groups
    
    def compute_curve_direction_PCA(self,points):
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigenvector with max eigenvalue
        principal_direction = eigvecs[:, np.argmax(eigvals)]
        return principal_direction
    
    def compute_average_curvature(self, points):
        if len(points) < 3:
            return 0
        curvatures = []
        for i in range(1, len(points) - 1):
            p0, p1, p2 = points[i-1], points[i], points[i+1] #curvature for p1
            v1 = p1 - p0
            v2 = p2 - p1
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0))
            curvature = angle / np.linalg.norm(v1) # angle per distance
            curvatures.append(curvature)
        return np.mean(curvatures)
    

    def compute_compactness_2d(self, points):
        if len(points) < 3:
            return 0
        centered = points - points.mean(axis=0)
        #singular value decomposition
        # rows of vh are vectors -> principal directions of the point cloud of the curve
        _, _, vh = np.linalg.svd(centered)
        # projection of points in 2 principal Directions
        projected = centered @ vh[:2].T

        try:
            hull = ConvexHull(projected)
            area = hull.volume
            #perimeter of convex hull
            hull_points = projected[hull.vertices]
            diffs = np.diff(np.vstack([hull_points, hull_points[0]]), axis=0)
            perimeter = np.sum(np.linalg.norm(diffs, axis=1))
            if perimeter == 0:
                return 0
            return area / (perimeter ** 2)
        except:
            return 0
 
if __name__ == "__main__":
    app =FeatureCurves()
    app.load_model("vase.ply")
    app.reset()
    app.mainLoop()
