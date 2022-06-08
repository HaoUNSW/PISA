from configparser import ConfigParser


class Options:
    '''
    This class handles the conversion of options set up in
    a config file to class attributes to facilitate parameter
    value access.
    '''

    def __init__(self, configFile):
        # set up all the class attributes from the configFile
        self.config = ConfigParser()
        self.config.read(configFile)
        self.getAttribFromConfig()

    # ------------------------------------------------------------------

    def getAttribFromConfig(self):
        '''
        We assume that the configFile has the right format so that
        all the parameters needed by the program are there. To make
        it easier to access the class attributes, the section headings
        are ignored. This function converts the config dictionary
        read from the config file into class attributes. After calling
        this function, new class attributes would be created. Since
        all the values in the config file are character string, the
        function also takes care of converting them to appropriate
        data type (boolean, int, float, or string).
        '''
        for s in self.config.sections():
            for op in self.config.options(s):
                value = self.config[s][op]
                if value == 'None':
                    self.__setattr__(op, None)
                elif value == 'True':
                    self.__setattr__(op, True)
                elif value == 'False':
                    self.__setattr__(op, False)
                else:
                    try:
                        if not '.' in value:
                            # no decimal point. Maybe an int?
                            value = int(value)
                        else:
                            # maybe a float?
                            value = float(value)
                    except:
                        try:
                            # maybe a float in exponential form, like 1e-3?
                            value = float(value)
                        except:
                            # none of the above. Must be a string then.
                            pass
                    self.__setattr__(op, value)

    # ------------------------------------------------------------------

    def toString(self):
        '''
        Funtion to output the class attributes and their contents to a
        string.
        '''
        string = ''
        for s in self.config.sections():
            string += '[' + str(s) + ']\n'
            for op in self.config.options(s):
                value = self.__getattribute__(op)
                string += (op + ' = ' + str(value))
                if type(value) is bool:
                    string += ' (type: bool)\n'
                elif type(value) is int:
                    string += ' (type: int)\n'
                elif type(value) is float:
                    string += ' (type: float)\n'
                else:
                    string += ' (type: string)\n'
            string += '\n'
        return string

    # ------------------------------------------------------------------

    def copyAttrib(self, obj):
        '''
        Function to copy all the attributes in the current object (self)
        to the destination object obj. Note that if the object obj
        has any attributes of the same names, they would be overwritten.

        :param obj: the destination object.
        '''
        for s in self.config.sections():
            for op in self.config.options(s):
                value = self.__getattribute__(op)
                obj.__setattr__(op, value)