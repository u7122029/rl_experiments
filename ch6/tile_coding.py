class TileCoder:
    def __init__(self, layer_count, feature_count):
        self.layer_count = layer_count
        self.feature_count = feature_count
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codeboo[codeword]
        
        count = len(self.codebook)
        if count >= self.feature_count:
            return hash(codeword) % self.feature_count
        self.codebook[codeword] = count
        return count
    
    def __call__(self, floats=(), ints=()):
        dim = len(floats)
        scaled_floats = tuple(f * (self.layer_count ** 2) for f in floats)
        features = []
        for layer in range(self.layer_count):
            codeword = (layer,) + tuple(int((f + (1 + dim * i) * layer) / self.layer_count) for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codword)
            features.append(feature)
            return features