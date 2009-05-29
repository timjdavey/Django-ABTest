from django.contrib.auth.models import User
from models import *
from utils import *
import random


class ABTest:
    def __init__(self, title, variates, db_persistent=False, session_persistent=True, session_name='abtest', default=None, description=None, control=None, binary=True, sample_size=None, confidence=None):
        self.__title = title
        
        if not isinstance(variates,list):
            raise Exception('List of "variates" passed needs to be of type list')
        self.__variates = variates
        
        # db_persistent is greater than session_persistent
        if db_persistent and not session_persistent:
            raise Warning('"db_persistent" cannot be True if "session_persistent" is False')
        # db_persistent ensures the user sees the same variate across sessions (default False)
        self.__db_persistent = db_persistent
        # session_persistent ensures the user sees the same variate across one session (default True)
        self.__session_persistent = session_persistent

        # this is the name used to store the variate on the user in the session
        self.__session_name = session_name
        
        # description to be associated with the experiment in the db
        if description and not isinstance(description, str):
            raise Exception('"description" needs to be a str')
        self.__description = description
        
        # binary means that the experiment has 2 outcomes, success or fail (such as link clicked or not)
        # false is used where value can be other than 0 or 1, such as when measuring user invites or revenue
        self.__binary = binary
        
        # control is used for checking that which variates are significatly better
        if control:
            # control can be integar or variate instance
            if isinstance(control,int):
                 if control >= len(variates):
                    raise Exception('"control" index is out of variate range length')
                 self.__control = variates[control]
            elif control in variates:
                self.__control = control
            else:
                raise Exception('"control" must be the 0-index of the variates list, or an instance of a variate itself')
        
        # so that you can force the experiment for finish at a certain sample size
        if sample_size:
            if not isinstance(sample_size, int):
                raise Exception('"sample_size" must be an integar')
            self.__sample_size = sample_size
        
        # if default is set then on assign_experiment an instance in the db shall be created
        # this means it is possible to store all inactions as well as all actions (such as button clicks)
        if not default == None and not isinstance(default, float) and not isinstance(default, int):
            raise Exception('"default" value needs to be a number (int or float)')
        else:
            self.__default = default
        # if default is set then on assign_experiment an instance in the db shall be created
        # this means it is possible to store all inactions as well as all actions (such as button clicks)
        if confidence and not isinstance(confidence, float) and not isinstance(confidence, int):
            raise Exception('"confidence" value needs to be a number (int or float)')
        elif confidence <= 0 and confidence >= 1:
            raise Exception('"confidence" value needs to be between 0 and 1')
        else:
            self.__confidence = confidence
    
    
    def get_variate(self, request=None):
        """
        Given a request, returns the variate which has been saved in the session or db for that user with None if none
        """
        var = None
        # tries to get from session
        if self.__session_persistent:
            var = self.get_variate_from_session(request)
        # tries to get from db
        elif self.__db_persistent and not var:
            var = self.get_variate_from_db(request)
        # tries to get from object
        elif hasattr(self,'_ABTest__variate'):
            var = self.__variate
        return var
    
    
    def get_variate_from_session(self, request):
        """
        Returns the variate if exists in session
        """
        session = request.session
        if session.get(self.__session_name) and session[self.__session_name].has_key(self.__title):
            return session[self.__session_name][self.__title]['variate']
        return None
    
    
    def get_variate_from_db(self, request):
        """
        Returns the variate from the db if that user has been apart from this experiment before
        Note: the user must be logged in and not an AnonymousUser
        """
        mvrs = None
        user = request.user
        if isinstance(user, User): # ensures sure is logged in and is not a AnonymousUser
            mvrs = ABResult.objects.filter(experiment__title=self.__title,user=user).order_by('-datetime_of')
            if mvrs:
                return self.__variates[mvrs[0].variate]
        return None
    
    
    def get_variate_from_index(self, index):
        return self.__variates[index]
    
    
    def get_result_id(self):
        """
        Returns the id of an result, if it has been assigned
        """
        if hasattr(self, '_ABTest__mvr'):
            return self.__mvr.id
        return None
    
    
    def assign_variate_from_index(index):
        self.__variate = self.__variates[index]
        return self
    
    
    def clear_variate(self, request):
        """
        clears the variate from the session, so can run the experiment again
        """
        session = request.session
        if session.get(self.__session_name) and session[self.__session_name].has_key(self.__title):
            del request.session[self.__session_name][self.__title]
        return self
    
    
    def random_variate(self):
        """
        Returns a random choice from the variates
        """
        return random.choice(self.__variates)
    
    
    def index_of_variate(self, var):
        for i in xrange(len(self.__variates)):
            if var == self.__variates[i]:
                return i
        raise Exception('Variate not in experiment variates list')
    
    
    def assign_experiment(self, request=None):
        """
        Given a request, assigns an experiment to a user if one is not currently running
        Returns the variate chosen
        """
        var = self.get_variate(request)
        if var and self.__session_persistent:
            # if this experiment is already running then just return the variate
            return var
        else:
            if self.__session_persistent:
                # make a session dictionary if one does not already exist
                if not request.session.has_key(self.__session_name):
                    request.session[self.__session_name] = {}
                # store the variate
                mvt = request.session[self.__session_name]
                var = self.random_variate()
                mvt[self.__title] = { 'variate': var }
            else:
                # if just want the variate
                var = self.random_variate()
        
        # if a default is set, creates a default result to override later
        if hasattr(self,'_ABTest__default'):
            # Gets or Creates a new experiment
            mve, was_new = ABExperiment.objects.get_or_create(title=self.__title,variates_length=len(self.__variates))
            # only saves the result if the experiment has not finished
            if not mve.finished:
                if self.__description:
                    mve.description = self.__description
                    mve.save()
                # creates the object in the db
                mvr = ABResult.objects.create(experiment=mve,variate=self.index_of_variate(var),value=self.__default)
                self.__mvr = mvr
                if self.__session_persistent:
                    mvt[self.__title]['default'] = mvr
            
        # reassigns the session dict with new choice
        if self.__session_persistent:
            request.session[self.__session_name] = mvt
        
        return var
    
    
    def assign_result(self, request=None, value=None, result=None):
        """
        Saves the result of an indiviual test of a user to the db with a value outcome (0-or-1 for fail-or-success)
        """
        if not value:
            raise Exception('Please provide a value')
        if request and result: 
            raise Exception('Please provide only a request or a result')
        if self.__binary and not value in [0,1]:
            raise Exception('Experiment is set to "binary=True", meaning only 0,1 or True,False as value maybe passed. Please set "binary=False" when initializing object if value is correct')

        # Create an experiment if one if not already running
        mve, was_new = ABExperiment.objects.get_or_create(title=self.__title,variates_length=len(self.__variates))
        # Gets the variate
        var = self.get_variate(request)
        
        # save the result only if the experiment has not finished
        if not mve.finished:
            if not var in self.__variates:
                raise Warning('"variate" from request.session is not in experiment')
            
            # if there was a default value, then find the initial result from creation
            if hasattr(self,'_ABTest__default') and self.__session_persistent:
                mvr = request.session[self.__session_name][self.__title]['default']
            elif result:
                mvr = ABResult.objects.get(pk=result)
            else:
                mvr = ABResult(experiment=mve,variate=self.index_of_variate(var))
            # assigns and saves
            mvr.value = value
            mvr.save()
            # Makes sure that the user is a proper instance of a User before trying to save to db
            if isinstance(request.user, User):
                mvr.user = request.user
            mvr.save()
            
        # saves the result in the session
        if self.__session_persistent:
            mvt = request.session[self.__session_name]
            mvt[self.__title]['result'] = value
            mvt[self.__title]['default'] = mvr
            request.session[self.__session_name] = mvt
        return var
    
    
    def has_result(self, request):
        """
        Returns whether the experiment has been run on this user already
        """
        s = request.session
        return s.has_key(self.__session_name) and s[self.__session_name].has_key(self.__title) and s[self.__session_name][self.__title].has_key('result')
    
    
    def stats(self, mvrs=None):
        """
        Returns the stats for the experiment - for outputting how you wish
        """
        if not mvrs:
            mvrs = ABResult.objects.filter(experiment__title=self.__title)
        stats = {}
        # for every variate result set
        for i in xrange(len(self.__variates)):
            results = mvrs.filter(variate=i)
            s = {'sum': 0, 'sum2': 0 }
            # calculate simple values
            for r in results:
                s['sum'] += r.value
                s['sum2'] += r.value**2
            if results:
                n = len(results)
                s['mean'] = s['sum']/n
                s['stddev'] = (s['sum2']/n - s['mean']**2)**0.5
                s['stderr'] = s['stddev']/(n**0.5)
            # assign results
            s['variate'] = self.__variates[i]
            stats[i] = s
        return stats
    
    
    def __gtest(self, variates, is_yates=True):
        """
        Is used to determine if there is a significant difference between
        two test results which are binary (success or fail experiments)
        Based on chi-squared null-hypothesis testing
        http://en.wikipedia.org/wiki/G-test
        
        largely ported from js at
        http://elem.com/~btilly/effective-ab-testing/g-test-calculator.html
        """
        
        try:
            # use the scipy chi-squared if available
            from scipy.stats import chi2
        except:
            # otherwise use my ported version
            from utils import chisqrprob
        else:
            chisqrprob = lambda x,y: 1-chi2.cdf(x,y)
        
        # sets up the data
        data = [[var['success'], var['expected']] for var in variates]
        len_var = len(variates)        
        sorted_data = variates
        sorted_data.sort(lambda x,y: int(x['value'])-int(y['value']))
        # works out control using given control or one with lowest value
        if hasattr(self,'_ABTest__control'):
            control = filter(lambda x: x['index']==self.__control, sorted_data)
        else:
            if sorted_data[0]:
                control = sorted_data[0]
            else:
                return [], 0
        # calculates the g-value and p-value for all the data as a full set
        g = self.__calculate_g_test(data, is_yates)
        p = chisqrprob(g, len_var-1)
        c = round(100*(1-p), 2)        
        # if there are simply two variates then the above g-values holds against conventional equation and so can simply return
        if len_var == 2:
            # return just the one winner
            winners = [(sorted_data[-1]['index'], c)]
        # with multiple variates compare all the variates against the control
        else:
            winners = []
            for sd in sorted_data:
                if not sd == control:
                    # re-run for each against the control
                    data = [ [control['success'],control['expected']] , [sd['success'],sd['expected']] ]
                    g = self.__calculate_g_test(data, is_yates)
                    p = chisqrprob(g, len_var-1)
                    
                    # need to add conversion factor to compensate for added variates
                    if len_var == 3:
                        p *= 1.65
                    elif len_var == 4:
                        p *= 2.8
                    elif len_var == 5:
                        p *= 5
                    elif len_var == 6:
                        p *= 8
                    else:
                        p *= len_var*(len_var -1)/2
                    
                    # only want the winners
                    if p < 1:
                        c = round(100*(1-p),2)
                        winners.append( (sd['index'], c) )
                        
        # want to return winners at the top
        winners.reverse()
        return winners, control['index']
    
    
    def __calculate_g_test(self, data, is_yates=False):
        # initializes sub-totals assuming a square matrix
        rows = xrange(len(data))
        row_totals = [0 for i in rows]
        cols = xrange(len(data[0]))
        col_totals = [0 for i in cols]
        total = 0
        
        # calculating the totals for the row and column
        for i in rows:
            for j in cols:
                e = data[i][j]
                row_totals[i] += e
                col_totals[j] += e
                total += e
                
        # calculating the g test from each entry
        g = 0
        for i in rows:
            for j in cols:
                e = row_totals[i]*col_totals[j]/total
                s = data[i][j]
                if is_yates:
                    if e+0.5 < s:
                        s -= 0.5
                    elif e-0.5 > s:
                        s += 0.5
                    else:
                        s = e
                g += 2*s*math.log(s/e)
        return g
    
    
    def __ztest(self, variates):
        """
        This simply works out the confidence-level using an unbaised m0 (which is approximated)
        for comparison with the null hypothesis
        http://en.wikipedia.org/wiki/Z-test
        """
        sorted_data = variates
        sorted_data.sort(lambda x,y: x['mean'] - y['mean'])
        
        # gets the control to test against
        if hasattr(self,'_ABTest__control'):
            control = filter(lambda x: x['index']==self.__control, sorted_data)
        else:
            if sorted_data[0]:
                control = sorted_data[0]
            else:
                return [], 0
        
        # works out confidences against control
        mean0 = control['mean']
        winners = []
        for v in sorted_data:
            if not v == control:
                winners.append(( self.__calculate_z_test(mean0, v['mean'], v['stderr'])))
        return winners, control['index']
    
    
    def __calculate_z_test(self, mean0, mean, stderr):
        return ZTABLE[round(abs(float(mean)-float(mean0)/float(stderr)),3)]
    
    
    def winners(self, mvrs=None):
        """
        Returns a list of winners for the experiment with the top confidence first
        as ( [ (variate, confidence_against_control) ] , variate_used_as_control)
        """
        if not mvrs:
            mvrs = ABResult.objects.filter(experiment__title=self.__title)
        if self.__binary:
            variates = []
            # put data into useable format
            for i in xrange(len(self.__variates)):
                m = filter(lambda x: x.variate==i, mvrs)
                if m:
                    s = float(len(filter(lambda x: x.value==1.0, m))) # successes
                    r = float(len(m))  # results
                    if r > s and s:
                        variates.append({ 'index': i, 'success': s, 'expected': r-s, 'value': s/r })
            return self.__gtest(variates)
        else:
            variates = []
            # put data info useable format
            for i in xrange(len(self.__variates)):
                results = filter(lambda x: x.variate==i, mvrs)
                if results:
                    m = math.sum(results)
                    s = math.sum(map(lambda x: x**2, results))
                    n = len(results)
                    mean = m/n
                    stddev = math.sqrt(s/n - math.pow(m,2))
                    stderr = stddev/math.sqrt(n)
                    variates.append( {'index': i, 'mean': mean, 'stddev': stddev, 'stderr': stderr } )
            return self.__ztest(variates)
        
    
    
    def evaluate(self,sample_size=None,force=False,confidence=None):
        """
        Evaluates the winning results based on a series of parameters
        This is will the experiment automatically is nessisary
        """
        
        if not sample_size and not hasattr(self, '_ABTest__sample_size') and not force and not confidence and not hasattr(self, '_ABTest__confidence'):
            raise Exception('Please provide a minimum "sample_size", "confidence" level or "force" the experiment to resolve. Otherwise just use winners() to see results.')
        
        mve = ABExperiment.objects.get(title=self.__title)
        
        # if the experiment has not already finished
        if not mve.finished:
            mvrs = ABResult.objects.filter(experiment=mve)
            winner = None
            # determine if is binary or not and use the appropriate test
            winners, control = self.winners(mvrs)
            # if sample_size is over its maximum limit or is being forced to finish just take top result
            if winners and ( len(mvrs) > sample_size or force ):
                winner = winners[0]
            # if there is a confidence level, then filter results by only those greater than that value
            if hasattr(self, '_ABTest__confidence'):
                confidence = self.__confidence
            if confidence:
                winners = filter( lambda x: x[1] > confidence, winners )
                if len(winners) > 1:
                    winner = winners[0]
            # save in the experiment
            if winner:
                mve.winner = winner
                mve.finished = True
                mve.confidence = confidence
                mve.save()
        else:
            # simply return if winner already exists and experiment is over
            winner = mve.winner
        
        if winner:
            return self.__variates[winner]
        else:
            return None
    
