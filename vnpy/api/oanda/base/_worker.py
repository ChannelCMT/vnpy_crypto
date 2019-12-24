import logging
import types

from vnpy.trader.utils import LoggerMixin

from ..ioloop import BackroundEventLoopProxy

class AsyncApiWorker(LoggerMixin):
    def __init__(self, api):
        super(AsyncApiWorker, self).__init__()
        self._api = api
        self._ioloop = api.ioloop
        self._proxy = BackroundEventLoopProxy(self._ioloop)
        self.register()

    @property
    def api(self):
        return self._api

    @property
    def ioloop(self):
        return self._ioloop

    def is_running(self):
        return self._api.is_running()

    def register(self):
        if self.process_transaction != types.MethodType(AsyncApiWorker.process_transaction, self):
            self._api.register_transaction_handler(self.process_transaction)
        if self.process_tick != types.MethodType(AsyncApiWorker.process_tick, self):
            self._api.register_tick_handler(self.process_tick)
        if self.process_response != types.MethodType(AsyncApiWorker.process_response, self):
            self._api.register_response_handler(self.process_response)
        if self.process_cancel_order != types.MethodType(AsyncApiWorker.process_cancel_order, self):
            self._api.register_cancel_order_handler(self.process_cancel_order)

    def process_transaction(self):
        raise NotImplementedError

    def process_tick(self):
        raise NotImplementedError

    def process_response(self):
        raise NotImplementedError

    def process_cancel_order(self):
        raise NotImplementedError

    def start(self):
        pass

    def close(self):
        pass

    def _on_task_finished(self, task):
        """Gather info when task halt unexpectedly"""
        if task.cancelled():
            return 
        exception = task.exception()
        if exception:
            self._api.on_error(exception)

    def create_task(self, coro):
        task = self._proxy.create_task_threadsafe(coro)
        task.add_done_callback(self._on_task_finished)
        return task

    def cancel_task(self, task):
        return self._proxy.cancel_task_threadsafe(task)

    def log(self, msg, level=logging.INFO):
        return self._api.log(msg, level=level)
