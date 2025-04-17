from vosk import Model, KaldiRecognizer
import json
import numpy as np
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

class VOSKEvaluator:
    def __init__(self, model_path="models/vosk_1/vosk-model-small-ru-0.22"):
        """
        Инициализация модели VOSK
        :param model_path: путь к модели VOSK
        """
        self.model = Model(model_path)
        self.sample_rate = 16000  # VOSK работает с 16kHz аудио

    def evaluate_audio(self, reference_text, audio_segment):
        """
        Оценка качества распознавания аудио
        :param reference_text: эталонный текст
        :param audio_segment: аудио объект pydub
        """
        # Конвертируем аудио в нужный формат для VOSK
        audio = audio_segment.set_frame_rate(self.sample_rate).set_channels(1)
        raw_data = audio.raw_data

        # Инициализируем распознаватель
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)  # Включаем вывод временных меток слов

        # Распознаем аудио
        rec.AcceptWaveform(raw_data)
        result = json.loads(rec.FinalResult())
        recognized_text = result.get("text", "")

        # Вычисляем метрики
        metrics = self._calculate_metrics(reference_text, recognized_text)

        return {
            "reference": reference_text,
            "recognized": recognized_text,
            "metrics": metrics
        }

    def _calculate_metrics(self, reference, recognized):
        """
        Вычисление метрик качества распознавания
        """
        ref_words = reference.split()
        rec_words = recognized.split()

        # Word Error Rate (WER)
        wer = levenshtein_distance(reference, recognized) / max(len(reference), 1)

        # Accuracy (точность) на уровне слов
        correct_words = sum(1 for r, h in zip(ref_words, rec_words) if r == h)
        word_accuracy = correct_words / max(len(ref_words), 1)

        # Precision, Recall, F1
        tp = sum(1 for word in rec_words if word in ref_words)
        fp = len(rec_words) - tp
        fn = len(ref_words) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "wer": wer,
            "word_accuracy": word_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def evaluate_multiple(self, samples):
        """
        Оценка на множестве примеров
        :param samples: список кортежей (reference_text, audio_segment)
        :return: агрегированные метрики
        """
        results = []
        for ref, audio in tqdm(samples):
            results.append(self.evaluate_audio(ref, audio))

        # Агрегируем метрики
        metrics = ["wer", "word_accuracy", "precision", "recall", "f1"]
        aggregated = {m: np.mean([r["metrics"][m] for r in results]) for m in metrics}

        return {
            "individual_results": results,
            "aggregated_metrics": aggregated
        }