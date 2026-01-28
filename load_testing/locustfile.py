from locust import HttpUser, task, between
import random

class ArabicRAGUser(HttpUser):
    wait_time = between(2, 5)

    @task(4)
    def ask_arabic_question(self):
        """
        Load test for Arabic RAG using ARCD-style factual questions
        """

        questions = [
            "ما هي عاصمة فلسطين؟",
            "من هو أبو عبيدة؟",
            "من هو مصطفى محمود؟",
            "من هو جمال خاشقجي؟",
            "أين يقع المسجد الأقصى؟",
            "متى وقعت نكبة فلسطين؟",
            "ما هي عاصمة الدولة الأموية؟",
            "من هو صلاح الدين الأيوبي؟",
            "أين تقع القدس؟",
            "ما هي منظمة الأمم المتحدة؟"
        ]

        payload = {
            "question": random.choice(questions)
        }

        self.client.post(
            "/api/ask/",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/api/ask (Arabic RAG)"
        )

    @task(1)
    def health_check(self):
        """
        Light request to keep system balanced
        """
        self.client.get("/api/health/")
