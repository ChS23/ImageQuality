import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from logging import log


st.title("Image Quality Metrics")

st.write("Оценка схожести изображений с помощью метрик качества изображений")

if 'history' not in st.session_state:
    st.session_state.history = []

hide_label = (
    """
<style>
    div[data-testid="stFileDropzoneInstructions"]>div>span {
       visibility:hidden;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>span::after {
       content:"INSTRUCTIONS_TEXT";
       visibility:visible;
       display:block;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>small {
       visibility:hidden;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>small::before {
       content:"FILE_LIMITS";
       visibility:visible;
       display:block;
    }
</style>
""".replace("INSTRUCTIONS_TEXT", "Перетащите изображение сюда")
    .replace("FILE_LIMITS", "Максимальный размер файла: 200 МБ")
)

st.markdown(hide_label, unsafe_allow_html=True)


def is_api_available() -> bool:
    url = "http://api:8080/"

    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError:
        return False

    if response.status_code == 200:
        return True
    else:
        return False


if not is_api_available():
    st.write("API загружается...")

else:

    def loss_request(
            _image1, _image2, _selected, _selected_metric
    ):
        url = "http://api:8080/loss"

        files = {
            "image1": image1.getvalue(),
            "image2": image2.getvalue()
        }

        data = {
            "package": _selected,
            "metric": _selected_metric,
        }

        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            value = response.json()["value"]
            st.session_state.history.append({"metric": f'{_selected}.{_selected_metric}', "value": value})
            return value
        else:
            st.error("Ошибка при запросе вычисления метрики! " + str(response.status_code))


    @st.cache_data
    def get_metrics_list(package: str) -> list:
        url = "http://api:8080/metrics"

        data = {
            "package": package,
        }

        response = requests.post(url, data=data)

        if response.status_code == 200:
            return response.json()["metrics"]
        else:
            st.error("Ошибка при запросе списка метрик! " + str(response.status_code))


    @st.cache_data
    def get_packages_list() -> list:
        url = "http://api:8080/packages"

        response = requests.post(url)

        if response.status_code == 200:
            return response.json()["packages"]
        else:
            st.error("Ошибка при запросе списка пакетов! " + str(response.status_code))


    col1, col2 = st.columns(2)

    with col1:
        image1 = st.file_uploader(
            "Выберите первое фото:",
            accept_multiple_files=False,
            type=["png", "jpg", "jpeg", "bmp"],
            key="1",
            help="Максимальный размер файла: 20 МБ"
        )

    with col2:
        image2 = st.file_uploader(
            "Выберите второе фото:",
            accept_multiple_files=False,
            type=["png", "jpg", "jpeg", "bmp"],
            key="2",
            help="Максимальный размер файла: 20 МБ"
        )

    st.write("Выберите метрику:")

    m1, m2 = st.columns(2)

    with m1:
        selected = st.selectbox(
            "Выберите пакет, в котором находится метрика:",
            get_packages_list()
        )

    with m2:

        metric_list = get_metrics_list(selected)

        selected_metric = st.selectbox(
            "Выберите метрику:",
            metric_list
        )

    if st.button("Посчитать"):
        if image1 is None or image2 is None:
            st.write("Загрузите оба изображения!")

        elif Image.open(image1).size != Image.open(image2).size:
            st.write("Изображения должны быть одинакового размера!")

        else:
            st.write("Метрика:", selected_metric)
            st.write("Значение:")
            st.write(loss_request(image1, image2, selected, selected_metric))

    if st.session_state.history:
        st.divider()
        st.write("История результатов:")
        reversed_history = st.session_state.history[::-1]
        history_table = st.table(reversed_history)
