import numpy as np

def grid_read_from_file(filename):
    my_file = open(filename, "r+")
    content = my_file.read()
    content = content.rstrip('\n')
    my_file.seek(0)

    my_file.write(content)
    my_file.truncate()
    my_file.close()

    file = open(filename, 'r')
    rows = 1
    columns = 0
    count = 0
    all_states = []
    while 1:
        state = file.read(1)
        if state:
            all_states.append(state)
        count += 1
        if not state:
            count -= 1
            break
        if state == '\n':
            all_states.pop()
            rows += 1
            count -= 1
            continue

    columns = int(count / rows)
    A = np.array(all_states)
    A = A.reshape((rows, columns))
    file.close()
    return A, rows, columns