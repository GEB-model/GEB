from time import time


class TimingModule:
    def __init__(self, name):
        self.name = name
        self.times = [time()]
        self.split_names = []

    def new_split(self, name):
        self.times.append(time())
        self.split_names.append(name)

    def __str__(self):
        messages = []
        for i in range(1, len(self.times)):
            time_difference = self.times[i] - self.times[i - 1]
            messages.append(
                "{}: {:.2f}s".format(self.split_names[i - 1], time_difference)
            )

        # Calculate total time
        total_time = self.times[-1] - self.times[0]
        messages.append("Total: {:.2f}s".format(total_time))

        to_print = ", ".join(messages)
        to_print = "{} - {}".format(self.name, to_print)

        return to_print
