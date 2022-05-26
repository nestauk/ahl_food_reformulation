# Scripts to save altair charts

from altair_saver import save
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import os

import ahl_food_reformulation
from ahl_food_reformulation import PROJECT_DIR


FIG_PATH = f"{PROJECT_DIR}/outputs/figures"

# Checks if the right paths exist and if not creates them when imported
os.makedirs(f"{FIG_PATH}/png", exist_ok=True)
os.makedirs(f"{FIG_PATH}/html", exist_ok=True)


def google_chrome_driver_setup():
    # Set up the driver to save figures as png
    driver = webdriver.Chrome(ChromeDriverManager().install())
    return driver


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
    save(fig, f"{path}/svg/{name}.svg", method="selenium", webdriver=driver)


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


if __name__ == "__main__":
    google_chrome_driver_setup()