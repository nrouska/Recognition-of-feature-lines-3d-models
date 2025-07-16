# Recognition-of-feature-lines-3d-models
Input: 3D meshes with known connectivity
1.Sort all points of a 3D shape according to whether they are on a vertex, edge, or flat area.
For this specific query, the eigenvalues of the covariance matrix of small patches can be analyzed and classified based on these values.

2.Use the previous classification to find features
lines (feature curves) defined e.g. from continuous edge points. For
for example, in a 3D face the points that define the outline of an eye
represent a characteristic line.

3.In the case that in a 3D shape there are more than one distinct
feature lines appear in a different color. Eventually we will
as many colors as the different characteristic lines that have been found.

4.In the event that two most distinct characteristic lines of a 3D
model, have common features, e.g. same shape, area, surface, distribution
normals, to be recognized as an object. For example the two
characteristic lines representing the outline of two eyes in one
person, to be recognized as belonging to the same class.

5. Find the similarity percentage of different models based on these
characteristics you have identified in a previous query. For example if
two human faces are given to be recognized as belonging to the same class.
