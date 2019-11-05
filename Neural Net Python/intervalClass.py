from datetime import date, time, datetime, timedelta

class IntervalClass:
      
    @staticmethod
    def secondsFromTime(time1):
        return time1.total_seconds()
       
    @staticmethod
    def stringToDateTime(dtString):
        if len(dtString) == 19:
            dt = datetime.strptime(dtString, "%Y-%m-%d %H:%M:%S")
        else:        
            dt = datetime.strptime(dtString, "%Y-%m-%d %H:%M:%S.%f")
        return dt
    
    @staticmethod
    def getHourFromDTString(dtString):
        dt = IntervalClass.stringToDateTime(dtString)
        return dt.hour
    
    @staticmethod
    def stringToTime(dtString):
        if len(dtString) == 8:
            dt = datetime.strptime(dtString, "%H:%M:%S")
        else:        
            dt = datetime.strptime(dtString, "%H:%M:%S.%f")
        return dt
    
    @staticmethod
    def calculateInterval(datetimeStr11, datetimeStr2):
        time1 = IntervalClass.stringToDateTime(datetimeStr2) - IntervalClass.stringToDateTime(datetimeStr11)
        return IntervalClass.secondsFromTime(time1)
        