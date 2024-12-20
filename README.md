# CU-Maths-Assignment-2_-Gram-Schmidt
import numpy as np

class GramSchmidt:
    def __init__(self, vectors):
        """
        Initializes the GramSchmidt class with input vectors after validation.
        """
        self.vectors = self.validate_vectors(vectors)

    def validate_vectors(self, vectors):
        """
        Validates a set of vectors for consistency and requirements.
        """
        vectors = [np.array(vec) for vec in vectors]

        if not vectors:
            raise ValueError("Input vector list is empty.")

        dims = [len(vec) for vec in vectors]
        if len(set(dims)) > 1:
            raise ValueError("All vectors must have the same number of elements.")

        if dims[0] < 10:
            raise ValueError("All vectors must have at least 10 elements.")

        if dims[0] > 10:
            print("Warning: Vectors have more than 10 elements. Using the first 10 elements.")
            vectors = [vec[:10] for vec in vectors]

        for vec in vectors:
            if not np.issubdtype(vec.dtype, np.number):
                raise ValueError("All vector elements must be numeric.")

        for i, vec in enumerate(vectors):
            if np.linalg.norm(vec) == 0:
                raise ValueError(f"Vector {i+1} is a zero vector.")

        seen = set()
        for i, vec in enumerate(vectors):
            vec_tuple = tuple(vec.tolist())
            if vec_tuple in seen:
                raise ValueError(f"Duplicate vector found at index {i+1}.")
            seen.add(vec_tuple)

        # Check for linear independence
        matrix = np.stack(vectors)
        rank = np.linalg.matrix_rank(matrix)
        if rank < len(vectors):
            raise ValueError("The input vectors are not linearly independent.")

        return vectors
