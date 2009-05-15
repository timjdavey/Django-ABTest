from django.contrib import admin
from models import *


class ABExperimentAdmin(admin.ModelAdmin):
    list_display = ('title','description','winner','variates_length','confidence','created_at','updated_at','finished')
    list_filter = ('finished',)
    search_fields = ('title',)
admin.site.register(ABExperiment,ABExperimentAdmin)

class ABResultAdmin(admin.ModelAdmin):
    list_display = ('experiment','variate','value','user','datetime_of')
    list_filter = ('experiment','variate',)
admin.site.register(ABResult,ABResultAdmin)
