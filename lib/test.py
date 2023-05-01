class Root:
    @property
    def models_path_gdrive(self):
        return self._models_path_gdrive
    @models_path_gdrive.setter
    def models_path_gdrive(self,value):
        return self._models_path_gdrive = value


print(f"static: Root.models_path_gdrive: {Root.models_path_gdrive}")
r = Root()
r.models_path_gdrive = "test"
print(f"object: r.models_path_gdrive: {r.models_path_gdrive}")

print(f"static again: Root.models_path_gdrive: {Root.models_path_gdrive}")