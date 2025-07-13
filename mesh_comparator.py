from feature_extractor import FeatureCurves
import numpy as np

class MeshComparator:
    def __init__(self, path1, path2):
        self.modelA = FeatureCurves()
        self.modelB = FeatureCurves()
        self.modelA.model = self.modelA.load_model(path1)
        self.modelB.model = self.modelB.load_model(path2)

        self.modelA.reset()
        self.modelB.reset()

    def compare_models(self, featuresA, featuresB, threshold=0.9):
        def one_way_score(source, target):
            #counter of source curves that match with target curves
            matches = 0
            #for each curve of source find similarity with every curve of target
            for vecA in source:
                similarities = [self.pearson_correlation(vecA, vecB) for vecB in target]
                # if the best similarity is over threshold there is a match
                if similarities and max(similarities) >= threshold:
                    matches += 1
             # returns percent of source curves that match with target curves
            return matches / len(source) if len(source) > 0 else 0

        score_AB = one_way_score(featuresA, featuresB)
        score_BA = one_way_score(featuresB, featuresA)
        return (score_AB + score_BA) / 2
    
    def pearson_correlation(self, vec1, vec2):
        return np.corrcoef(vec1, vec2)[0, 1]

    def run(self):
        featuresA = self.modelA.features
        featuresB = self.modelB.features
        similarity = self.compare_models(featuresA, featuresB)
        print(f"Similarity score: {similarity * 100:.2f}%")

if __name__ == "__main__":
  
    model_path1 = "portrait1.ply"
    model_path2 = "portrait2.ply"
    comparator = MeshComparator(model_path1, model_path2)
    comparator.run()