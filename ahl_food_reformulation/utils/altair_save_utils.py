# %% [markdown]
# Scripts to save altair charts

# %%
from altair_saver import save
from altair import Chart
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import os

# %%
import ahl_food_reformulation
from ahl_food_reformulation import PROJECT_DIR


# %%
FIG_PATH = f"{PROJECT_DIR}/outputs/figures"

# %%
# Checks if the right paths exist and if not creates them when imported
os.makedirs(f"{FIG_PATH}/png", exist_ok=True)
os.makedirs(f"{FIG_PATH}/html", exist_ok=True)


# %%
def google_chrome_driver_setup():
    # Set up the driver to save figures as png
    driver = webdriver.Chrome(ChromeDriverManager().install())
    return driver


# %%
def save_altair(fig, name, driver, path=FIG_PATH):
    """Saves an altair figure as png and html
    Args:
        fig: altair chart
        name: name to save the figure
        driver: webdriver
        path: path to save the figure
    """
    # Save png
    save(
        fig,
        f"{path}/png/{name}.png",
        method="selenium",
        webdriver=driver,
        scale_factor=5,
    )
    # Save html
    fig.save(f"{path}/html/{name}.html")
    # save svg
    # save(fig, f"{path}/svg/{name}.svg", method="selenium", webdriver=driver)


# %%
def save_altair_to_path(
    fig, name, driver, path=FIG_PATH, save_png=False, save_html=True
):
    """Saves an altair figure as png and html
    Args:
        fig: altair chart
        name: name to save the figure
        driver: webdriver
        path: path to save the figure
    """
    # Save png
    if save_png:
        save(
            fig,
            f"{path}/{name}.png",
            method="selenium",
            webdriver=driver,
            scale_factor=5,
        )
    if save_html:
        # Save html
        fig.save(f"{path}/{name}.html")


def altair_text_resize(chart: Chart, sizes: tuple = (12, 14)) -> Chart:
    """Resizes the text of axis labels and legends in an altair chart
    Args:
        chart: chart to resize
        sizes: label size and title size
    Returns:
        An altair chart
    """

    ch = chart.configure_axis(
        labelFontSize=sizes[0], titleFontSize=sizes[1], labelLimit=300
    ).configure_legend(labelFontSize=sizes[0], titleFontSize=sizes[1])
    return ch


# Scripts to save altair charts

# import json
# import os

# import altair as alt
# from altair import Chart
# from altair_saver import save
# from selenium import webdriver
# from selenium.webdriver.chrome.webdriver import WebDriver
# from webdriver_manager.chrome import ChromeDriverManager


# from ahl_food_reformulation import PROJECT_DIR


# FIG_PATH = f"{PROJECT_DIR}/outputs/figures"

# # Checks if the right paths exist and if not creates them when imported
# os.makedirs(f"{FIG_PATH}/png", exist_ok=True)
# os.makedirs(f"{FIG_PATH}/html", exist_ok=True)


# def google_chrome_driver_setup() -> WebDriver:
#     # Set up the driver to save figures as png
#     driver = webdriver.Chrome(ChromeDriverManager().install())
#     return driver


# def save_altair(fig: Chart, name: str, driver: WebDriver, path: str = FIG_PATH) -> None:
#     """Saves an altair figure as png and html
#     Args:
#         fig: altair chart
#         name: name to save the figure
#         driver: webdriver
#         path: path to save the figure
#     """
#     save(
#         fig,
#         f"{path}/png/{name}.png",
#         method="selenium",
#         webdriver=driver,
#         scale_factor=5,
#     )
#     fig.save(f"{path}/html/{name}.html")


# def altair_text_resize(chart: Chart, sizes: tuple = (12, 14)) -> Chart:
#     """Resizes the text of axis labels and legends in an altair chart
#     Args:
#         chart: chart to resize
#         sizes: label size and title size
#     Returns:
#         An altair chart
#     """

#     ch = chart.configure_axis(
#         labelFontSize=sizes[0], titleFontSize=sizes[1], labelLimit=300
#     ).configure_legend(labelFontSize=sizes[0], titleFontSize=sizes[1])
#     return ch


# def make_save_path(path: str):
#     """Make save paths in case we are not using
#     the standard one
#     """

#     os.makedirs(f"{path}/png", exist_ok=True)
#     os.makedirs(f"{path}/html", exist_ok=True)


# %%
if __name__ == "__main__":
    google_chrome_driver_setup()
