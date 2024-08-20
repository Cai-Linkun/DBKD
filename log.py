import logging
import sys

console_handler = logging.StreamHandler(stream=sys.stdout)
file_handler = logging.FileHandler("app.log", delay=0, encoding="utf8")
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, console_handler],
)

logger = logging.getLogger(__name__)
