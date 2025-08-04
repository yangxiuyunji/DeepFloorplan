try:
    import demo
    print('Demo imported successfully')
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()
