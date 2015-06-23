from sys import stderr
import locale

class Counter(object):
    def __init__(self, step=10):
        self.cur = 0
        self.step = step

    def tick(self):
        self.cur += 1
        if self.cur % self.step == 0:
            stderr.write( str(self.cur) )
            stderr.write( "\r" )
            stderr.flush()
        
    def done(self):
        stderr.write( str(self.cur) )
        stderr.write( "\n" )
        stderr.flush()

class CommaCounter(Counter): #requires python 2.7
    def tick(self):
        self.cur += 1
        if self.cur % self.step == 0:
            stderr.write( '{:,}'.format(self.cur) )
            stderr.write( "\r" )
            stderr.flush()
    def done(self):
        stderr.write( '{:,}'.format(self.cur) )
        stderr.write( "\n" )
        stderr.flush()

class Progress(object):
    def __init__(self, numSteps):
        self.total = numSteps
        self.cur = 0
        self.curPercent = 0
    def tick(self):
        self.cur += 1
        newPercent = (100*self.cur)/self.total
        if newPercent > self.curPercent:
            self.curPercent = newPercent
            stderr.write( str(self.curPercent)+"%" )
            stderr.write( "\r" )
            stderr.flush()
    def done(self):
        stderr.write( '100%' )
        stderr.write( "\n" )
        stderr.flush()

def ProgressLine(line):
    stderr.write(line)
    stderr.write( "\r" )
    stderr.flush()
    
def main():
    from time import sleep
    for i in range(500):
        s = str(2.379*i)
        ProgressLine(s)
        sleep(0.02)
    c = Counter(5)
    for i in range(500):
        c.tick()
        sleep(.005)
    c.done()
    p = Progress(5000)
    for i in range(5000):
        p.tick()
        sleep(.0005)
    p.done()


if __name__ == "__main__":
    main()
    
