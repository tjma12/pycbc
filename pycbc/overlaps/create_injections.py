#! /usr/bin/env python

def get_outfilename(output_directory, ifos_tag = 'HL', user_tag = None, gz = True):
    tag = ''
    if user_tag is not None:
        tag = '_%s' % user_tag
    extension = 'xml'
    if gz:
        extension += '.gz'
    return '%s/%s-INJECTIONS%s.%s' % (output_directory, ifos_tag, tag, extension)

#
#   Some Useful Functions
#
def getCompMasses(mtotal, q):
    """
    Returns mass1, mass2 from the total mass and mass ratio, where
    mass1 >= mass2 and q = mass1/mass2.
    """
    return float(q*mtotal)/ (1+q), float(mtotal)/ (1+q)

def getEta(mass1, mass2):
    """
    Returns the symmetric mass ratio using the component masses as input.
    """
    return (mass1*mass2)/ float((mass1+mass2)**2.)

def getMchirp(mass1, mass2):
    """
    Returns mchirp using the total mass and mass ratio as input.
    """
    return getEta(mass1, mass2)**(3./5.) * (mass1 + mass2)

