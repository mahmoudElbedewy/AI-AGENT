import re
from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

class StrictRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}), label='الباسوورد')
    confirm_password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}), label='تأكيد الباسوورد')

    class Meta:
        model = User
        fields = ['username']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'})
        }

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if not re.match(r'^[a-zA-Z0-9_]{5,20}$', username):
            raise ValidationError("اليوزرنيم لازم يكون من 5 لـ 20 حرف أو رقم إنجليزي فقط، ومفيش مسافات.")
        
        if User.objects.filter(username=username).exists():
            raise ValidationError("اليوزرنيم ده مستخدم قبل كده، جرب واحد تاني.")
        
        return username

    def clean_password(self):
        password = self.cleaned_data.get('password')
        if len(password) < 8:
            raise ValidationError("الباسوورد لازم يكون 8 حروف على الأقل.")
        if not re.search(r'[A-Z]', password):
            raise ValidationError("الباسوورد لازم يحتوي على حرف كابيتال واحد على الأقل.")
        if not re.search(r'[a-z]', password):
            raise ValidationError("الباسوورد لازم يحتوي على حرف سمول واحد على الأقل.")
        if not re.search(r'[0-9]', password):
            raise ValidationError("الباسوورد لازم يحتوي على رقم واحد على الأقل.")
        if not re.search(r'[@$!%*?&#]', password):
            raise ValidationError("الباسوورد لازم يحتوي على رمز خاص (زي @$!%*?&#).")
        return password

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        # التأكد إن الباسوورد متطابق
        if password and confirm_password and password != confirm_password:
            self.add_error('confirm_password', "الباسوورد وتأكيده مش متطابقين.")

