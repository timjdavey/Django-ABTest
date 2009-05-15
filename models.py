from django.contrib.auth.models import User
from django.db import models


class ABExperiment(models.Model):
    """
    This the meta model which stores the information for each experiment
    """
    created_at = models.DateTimeField(auto_now_add=1)
    updated_at = models.DateTimeField(auto_now=1)
    
    title = models.CharField(max_length=200)
    description = models.TextField(null=True,blank=True)
    variates_length = models.PositiveIntegerField(null=False)
    
    winner = models.PositiveIntegerField(null=True)
    finished = models.BooleanField(null=True,default=False,db_index=True)
    confidence = models.FloatField(null=True)

    def __unicode__(self):
        if not hasattr(self,'pk'): self.pk = 'X'
        return '%s. %s' % (str(self.pk), self.title)


class ABResult(models.Model):
    """
    An instance of the case
    """
    datetime_of = models.DateTimeField(auto_now_add=1)
    
    experiment = models.ForeignKey(ABExperiment)
    user = models.ForeignKey(User, null=True)
    
    variate = models.PositiveIntegerField(null=False)
    value = models.FloatField(null=False)

    def __unicode__(self):
        if not hasattr(self,'pk'): self.pk = 'X'
        return '%s. for %s' % (str(self.pk), str(self.experiment_id))