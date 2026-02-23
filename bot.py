import logging

from telegram.ext import Application
from telegram.request import HTTPXRequest

from tilesheet_bot.background_remover import RemoveBgBackgroundRemover
from tilesheet_bot.bot_controller import BotController
from tilesheet_bot.config import BotConfig
from tilesheet_bot.session_manager import SessionManager
from tilesheet_bot.storage import TempDirStorage


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = BotConfig.from_env()

    session_manager = SessionManager()
    storage = TempDirStorage(config.temp_dir)
    background_remover = RemoveBgBackgroundRemover()
    controller = BotController(
        session_manager,
        background_remover,
        storage,
        max_images_per_session=config.max_images_per_session,
        debug_postprocess_dir=(
            config.debug_postprocess_dir if config.debug_postprocess else None
        ),
        send_read_timeout=config.send_read_timeout,
        send_write_timeout=config.send_write_timeout,
        send_connect_timeout=config.send_connect_timeout,
        send_pool_timeout=config.send_pool_timeout,
    )

    request = HTTPXRequest(
        read_timeout=config.read_timeout,
        write_timeout=config.write_timeout,
        connect_timeout=config.connect_timeout,
        pool_timeout=config.pool_timeout,
    )
    app = Application.builder().token(config.token).request(request).build()
    controller.register(app)
    app.run_polling()


if __name__ == "__main__":
    main()
