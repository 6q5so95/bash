import win32event
import win32api

class MutexManager:
    mutex_name = 'MyProgramMutex'

    @classmethod
    def acquire_lock(cls, caller_package):
        mutex_handle = None
        try:
            # Mutexを作成し、取得を試みる
            mutex_handle = win32event.CreateMutex(None, 1, cls.mutex_name)
        except win32api.error as e:
            if e.winerror == 5:  # ERROR_ACCESS_DENIED
                print("Mutexが既に取得されています。")
                return None
            else:
                raise

        # ここに必要な処理を書く

        return mutex_handle

    @classmethod
    def release_lock(cls, mutex_handle, caller_package):
        if mutex_handle:
            # Mutexを解放する
            win32event.CloseHandle(mutex_handle)
            print(f"Mutexが解放されました。呼び出し元のパッケージ: {caller_package}")

# 利用例
if __name__ == "__main__":
    # ロックが取得できない場合の処理
    mutex_handle = MutexManager.acquire_lock(__package__)
    if not mutex_handle:
        # ロックが既に取得されていた場合の処理
        print("プログラムが既に実行中です.")
    else:
        # ロックが取得できた場合の処理
        print("Mutexが取得されました.")

        try:
            # ここにアプリコードを書く
            pass
        finally:
            # Mutexを解放する
            MutexManager.release_lock(mutex_handle, __package__)
