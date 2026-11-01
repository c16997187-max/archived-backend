import logging


class StripQueryString(logging.Filter):
    """Keep query strings out of Uvicorn's access log: `?q=` holds students' names."""

    # uvicorn logs: '%s - "%s %s HTTP/%s" %d' (client, method, path?query, version, status)
    _PATH_ARG = 2

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) > self._PATH_ARG:
            path = args[self._PATH_ARG]
            if isinstance(path, str):
                record.args = (
                    *args[: self._PATH_ARG],
                    path.partition("?")[0],
                    *args[self._PATH_ARG + 1 :],
                )
        return True


_strip_query_string = StripQueryString()


def install_log_filters() -> None:
    # One shared instance: addFilter ignores it when create_app runs again.
    logging.getLogger("uvicorn.access").addFilter(_strip_query_string)
