import asyncio, sys, json
sys.path.insert(0, 'src')

async def test():
    import websockets
    async with websockets.connect('ws://localhost:8766') as ws:
        for i in range(8):
            msg = await asyncio.wait_for(ws.recv(), timeout=3)
            data = json.loads(msg)
            print(f"phase={data['phase']} yellow={data['yellow']} queue={data['queue']}")
            for a, vehs in data['vehicles'].items():
                positions = [round(v['position'], 2) for v in vehs]
                print(f"  {a}: {positions}")
            print()

asyncio.run(test())
