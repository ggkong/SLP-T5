from django.shortcuts import render
from subCellLoc.dpmodel import getFeatureT5, getClassifyModel
from subCellLoc.models import UserProfile
from subCellLoc.tools import handle_uploaded_file

# Create your views here.

def showHtml(request):
    return render(request, './sub_page.html')


def modelRun(request):
    name = request.POST.get("name")
    email = request.POST.get("email")
    organization = request.POST.get("organization")
    careers = request.POST.get("careers")
    new_user = UserProfile(name=name, email=email, organization=organization,careers=careers)
    new_user.save()
    data = request.POST.get("data")
    upload_str = handle_uploaded_file(request.FILES['file'])

    if data != "":
        upload_str = data.replace("\n", "")
    feature = getFeatureT5(upload_str)
    result = getClassifyModel(feature)

    if len(result) == 0:
        message = "The protein sequence you provided was not recognized by our system."
    elif len(result) == 1:
        message = "The protein sequence you provided has been predicted by our system as " + result[0] + "."
    else:
        result = ', '.join(result)
        final_result = f"'{result}'"
        message = "The protein sequence you provided has been predicted by our system as " + final_result + "."
    return render(request, './result_subcell.html', {'result_seq': message})
