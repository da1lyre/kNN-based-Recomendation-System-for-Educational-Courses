import numpy as np
import pandas as pd
from collections import Counter
import joblib
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


class Recommender:
    def __init__(self, data, df, knn, popular_courses):
        self.data = data
        self.df = df
        self.knn = knn
        self.popular_courses = popular_courses

    def get_user_vector(self, course_ids):
        if not course_ids:
            return np.zeros(self.data.shape[1])
        return np.mean(self.data[course_ids], axis=0)

    def find_similar_users(self, user_vector, k_neighbors=12):
        distances, indices = self.knn.kneighbors(
            user_vector.reshape(1, -1),
            n_neighbors=k_neighbors
        )
        return distances[0], indices[0]

    def get_recommendations(self, user_course_ids, top_n=8):
        if not user_course_ids:
            sampled = np.random.choice(
                self.popular_courses,
                size=min(top_n, len(self.popular_courses)),
                replace=False
            )
            results = self.df.iloc[sampled][
                ["course_title", "course_organization", "course_difficulty", "course_rating"]
            ].copy()
            results["source"] = "Popular (cold start)"
            results["score"] = 0
            return results

        user_vector = self.get_user_vector(user_course_ids)
        distances, indices = self.find_similar_users(user_vector, k_neighbors=12)

        neighbor_courses = []
        for idx in indices:
            neighbor_courses.append(idx)

        course_counter = Counter(neighbor_courses)

        for course_id in user_course_ids:
            if course_id in course_counter:
                del course_counter[course_id]

        knn_course_ids = [
            course_id for course_id, _ in course_counter.most_common(12)
            if course_id not in user_course_ids
        ][:7]

        used_indices = set(user_course_ids)
        used_indices.update(knn_course_ids)

        available_popular = [c for c in self.popular_courses if c not in used_indices]
        popular_selected = []
        if available_popular:
            popular_selected.append(np.random.choice(available_popular))

        all_course_ids = knn_course_ids + popular_selected

        if len(all_course_ids) < top_n:
            for course_id in self.popular_courses:
                if course_id not in used_indices:
                    all_course_ids.append(course_id)
                    used_indices.add(course_id)
                    if len(all_course_ids) >= top_n:
                        break

        results = self.df.iloc[all_course_ids][
            ["course_title", "course_organization", "course_difficulty", "course_rating"]
        ].copy()

        sources = []
        for i in range(len(all_course_ids)):
            if i < 7:
                sources.append("k-NN")
            else:
                sources.append("Popular (random)")

        results["source"] = sources
        results["score"] = [course_counter.get(course_id, 0) for course_id in all_course_ids]

        return results

    def recommend_by_titles(self, course_titles):
        if not course_titles:
            return self.get_recommendations([])

        course_ids = []
        for title in course_titles:
            matches = self.df[self.df["course_title"].str.contains(title, case=False)]
            if len(matches) > 0:
                course_ids.append(matches.index[0])

        if not course_ids:
            return self.get_recommendations([])

        return self.get_recommendations(course_ids)


def detect_language(text):
    if pd.isna(text) or text == "":
        return "unknown"
    try:
        return detect(text)
    except:
        return "unknown"

def load_model():
    artifacts = joblib.load("../models/knn_recommender_final.pkl")

    recommender = Recommender(
        data=artifacts["data"],
        df=df,
        knn=artifacts["knn_model"],
        popular_courses=artifacts["popular_courses"]
    )

    return recommender, df

df = pd.read_csv("../data/coursera_courses.csv")
df["language"] = df["course_title"].apply(detect_language)
df = df[df["language"] == "en"].copy()
df = df.drop(columns=["language"])
df = df.reset_index(drop=True)