from vvrpywork.constants import *
from vvrpywork.scene import *
from vvrpywork.shapes import *

from random import random

WIDTH = 800
HEIGHT = 800

class Project(Scene3D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Project")
        self.reset()

    def reset(self):
        self.model = Mesh3D("resources/dragon_low_low.obj", color=Color.GRAY)
        self.addShape(self.model, "model")
        self.model_w = LineSet3D.create_from_mesh(self.model)
        self.addShape(self.model_w,"wireframe")
        vc=self.model.vertex_colors

        self.adj_list = [find_adjacent_vertices(self.model, i) for i in range(len(self.model.vertices))]

        edges,corners,faces=self.principal_axes(self.model,self.adj_list)
        print(corners)
        
        vc[edges] = (0, 1, 0)   
        vc[corners] = (1, 0, 0) 
        vc[faces]= (0 , 0,1)
    
        self.model.vertex_colors = vc
        self.updateShape("model")

       

    def principal_axes(self,model:Mesh3D,adj_list:list):
        
        vertices = model.vertices
        normals = model.vertex_normals
        triangles = model.triangles
        vc = model.vertex_colors
        edges=[]
        corners=[]
        faces=[]

        
        for i in range(len(vertices)):
            adj = adj_list[i]

            normal_neigbors = normals[adj] - normals[adj].mean(axis=0)
            centered_neighbors=vertices[adj] - vertices[adj].mean(axis=0)

            covariance_matrix=np.cov(normal_neigbors,rowvar=False) #variables are in columns
            eigenvalues = np.linalg.eigvals(covariance_matrix)
            eigenvalues = np.sort(eigenvalues)
            sum_eigs = np.sum(eigenvalues)

            if sum_eigs < 1e-6:
                continue

            # Normalize eigenvalues
            l1, l2, l3 = eigenvalues / sum_eigs # l3>l2>l1 ## l1+l2+l3 = 1
            
            if l1/l3 > 0.08:  #  corner variation in all directions l3~l2~l1
               corners.append(i)
            elif l2/l1 > 5:  # face l3~l2 >> l1
                faces.append(i)
            elif l3/(l1 +l2) > 5  : # edge l3>> l1~l2
                edges.append(i)

        
        return edges,corners,faces
           
      
def find_adjacent_vertices(mesh: Mesh3D, idx: int) -> np.ndarray:

    vertices = mesh.vertices
    triangles = mesh.triangles
    adj=[]
    adjacent_triangles = triangles[np.any(triangles == idx, axis=1)]
    adj.extend(adjacent_triangles[adjacent_triangles != idx])
    return np.unique(adj)



if __name__ == "__main__":
    app = Project()
    app.mainLoop()