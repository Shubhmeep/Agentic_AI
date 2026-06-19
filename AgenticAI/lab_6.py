####################################################
# async python - very important 
####################################################


# synchronous python 
import time

def make_request(name):
    print(f"Starting {name}")
    time.sleep(2)
    print(f"Finished {name}")
    print()


# make_request("Request 1")
# make_request("Request 2")
# make_request("Request 3")

'''
This is synchronous sequential execution:

Do task 1 completely
Then task 2 completely
Then task 3 completely
'''

# Writing async def does not automatically make multiple operations concurrent.
import asyncio

async def do_some_work():
    print("Starting work")
    await asyncio.sleep(1)
    print("Work complete")
    return "done"

async def main():
    resp = await do_some_work()
    print(resp)

asyncio.run(main())


'''
Behind the scenes:

do_some_work() is a coroutine.
asyncio.run() starts an event loop.
When await asyncio.sleep(1) is hit, the event loop pauses do_some_work() and can run other coroutines.

'''

# Run multiple coroutines concurrently
import asyncio

async def task(name, delay):
    await asyncio.sleep(delay)
    print(f"{name} done after {delay}s")
    return name

async def main():
    results = await asyncio.gather(
        task("A", 1),
        task("B", 2),
        task("C", 3)
    )
    print(results)

asyncio.run(main())