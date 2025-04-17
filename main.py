import os
import random
from pydub import AudioSegment
from pydub.effects import normalize
import wave
from num2words import num2words

from vosk_1 import VOSKEvaluator


class GenerateAudio:

    def __init__(self):
        self.fraction = ['дроби', 'дробь']
        self.fraction_path = 'fractions'

        self.word_dom = ['дом']
        self.word_dom_path = 'words'

        self.word_kvartira = ['квартира']
        self.word_kvartira_path = 'words'

        self.alphabet = [elem.replace('.wav', '') for elem in os.listdir('alphabet')]
        self.alphabet_path = 'alphabet'

        self.numbers = [elem.replace('.wav', '') for elem in os.listdir('numbers')]
        self.numbers_path = 'numbers'

        self.streets = [elem.replace('.wav', '') for elem in os.listdir('street')]
        self.streets_path = 'street'

        self.pause_duration = 100

    def _load_audio(self, path, filename, default_ext='.wav'):
        """Загрузка аудиофайла"""
        audio = AudioSegment.from_file(os.path.join(path, filename + default_ext))
        return normalize(audio)

    def _add_pause(self, audio_segment, pause_duration=None):
        """Добавление паузы к аудио"""
        if pause_duration is None:
            if random.random() > 0.5:
                pause_duration = random.random() * 1000
            else:
                pause_duration = 100
        silence = AudioSegment.silent(duration=pause_duration)
        return audio_segment + silence


    def _change_speed_random(self, audio):
        """Изменение скорости аудио с сохранением тона"""

        # Преобразуем в сырые данные
        raw_data = audio.raw_data
        channels = audio.channels
        sample_width = audio.sample_width
        frame_rate = int(audio.frame_rate * random.randrange(5, 15, 1)/10)

        # Создаем новый аудиосегмент с измененной скоростью
        return AudioSegment(
            data=raw_data,
            sample_width=sample_width,
            frame_rate=frame_rate,
            channels=channels
        )

    def generate_address_audio(self, street, number_dom, number_kvartira, output_file='output.wav', save=False):
        """Генерация аудио с адресом"""
        # Начинаем с тишины
        combined = AudioSegment.silent(duration=100)

        # Добавляем улицу
        street_audio = self._load_audio(self.streets_path, street)
        if street_audio:
            combined = self._add_pause(combined) + street_audio

        # Обрабатываем номер дома
        dom_parts = number_dom.split('_')
        for part in dom_parts:
            # Определяем путь к файлу в зависимости от типа части
            if part in self.word_dom:
                audio = self._load_audio(self.word_dom_path, part)
            elif part in self.fraction:
                audio = self._load_audio(self.fraction_path, part)
            elif part in self.alphabet:
                audio = self._load_audio(self.alphabet_path, part)
            elif part in self.numbers:
                audio = self._load_audio(self.numbers_path, part)
            else:
                continue

            if audio:
                combined = self._add_pause(combined) + audio

        # Обрабатываем номер квартиры
        kvartira_parts = number_kvartira.split('_')
        for part in kvartira_parts:
            # Определяем путь к файлу в зависимости от типа части
            if part in self.word_kvartira:
                audio = self._load_audio(self.word_kvartira_path, part)
            elif part in self.fraction:
                audio = self._load_audio(self.fraction_path, part)
            elif part in self.alphabet:
                audio = self._load_audio(self.alphabet_path, part)
            elif part in self.numbers:
                audio = self._load_audio(self.numbers_path, part)
            else:
                continue

            if audio:
                combined = self._add_pause(combined) + audio

        # Экспортируем результат
        if random.random() > 0.6:  # Изменяем скорость
            combined = self._change_speed_random(combined)
        if save:
            combined.export(output_file, format="wav")
            return output_file
        return combined

    def get_sample(self):
        street = random.choice(self.streets)

        number_dom = random.choice(self.numbers)
        if random.random() > 0.5:   # Добавляем слово дом
            number_dom = random.choice(self.word_dom) + "_" + number_dom
        if random.random() > 0.8:  # Добавляем дробь для дома
            number_dom = number_dom + "_" + random.choice(self.fraction) + "_" + random.choice(
                self.alphabet + self.numbers)

        number_kvartira = random.choice(self.numbers)
        if random.random() > 0.5:   # Добавляем слово квартира
            number_kvartira = random.choice(self.word_kvartira) + "_" + number_kvartira
        if random.random() > 0.8:   # Добавляем дробь для квартиры
            number_kvartira = number_kvartira + "_" + random.choice(self.fraction) + "_" + random.choice(
                self.alphabet + self.numbers)

        return street, number_dom, number_kvartira

    def generate_random_address_audio(self, save=False):
        """Генерация случайного адреса и его аудио"""
        street, dom, kvartira = self.get_sample()
        output_file = "_".join([street, dom, kvartira]) + ".wav"

        text = " ".join([street, dom, kvartira])
        text = text.replace('_', ' ')
        numbers = [word for word in text.split() if word.isdigit()]
        for number in numbers:
            text = text.replace(number, num2words(int(number), lang='ru'))

        return text.lower(), self.generate_address_audio(street, dom, kvartira, output_file, save)

vosk_1 = VOSKEvaluator()

audio_generator = GenerateAudio()
results = []
batch = []
for _ in range(0, 1000):
    batch.append(audio_generator.generate_random_address_audio())

result = vosk_1.evaluate_multiple(batch)
print(f"WER: {result.get('aggregated_metrics')['wer']}")
print(f"WER: {result.get('aggregated_metrics')['word_accuracy']}")
print(f"WER: {result.get('aggregated_metrics')['precision']}")
print(f"WER: {result.get('aggregated_metrics')['f1']}")