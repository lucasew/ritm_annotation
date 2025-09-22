import dagger
from dagger import dag, function, object_type


@object_type
class RitmAnnotation:
    @function
    def gettext_container(self) -> dagger.Container:
        return (
            dag.container()
            .from_("alpine:latest")
            .with_exec(["apk", "add", "gettext", "bash", "python3"])
        )

    @function
    async def codegen_fixes(self, source: dagger.Directory) -> dagger.Directory:
        output = source
        output = await self.update_locales(output)
        return output

    @function
    async def update_locales(self, source: dagger.Directory) -> dagger.Directory:
        return (
            await self.gettext_container()
            .with_mounted_directory("/src", source)
            .with_workdir("/src")
            .with_exec(["./update_locales"])
            .directory("/src")
        )

    @function
    async def test_update_date(self, source: dagger.Directory):
        return (
            await dag.container()
            .from_("alpine:latest")
            .with_mounted_directory("/src", source)
            .with_exec(["/bin/sh", "-c", "date > /src/date.txt"])
            .with_exec(["ls", "/src"])
            .directory("/src")
            # .export(".") # you need to run export in the cli
        )

    # @function
    # def update_locale(self, locale: str):

    @function
    def container_echo(self, string_arg: str) -> dagger.Container:
        """Returns a container that echoes whatever string argument is provided"""
        return dag.container().from_("alpine:latest").with_exec(["echo", string_arg])

    @function
    async def grep_dir(self, directory_arg: dagger.Directory, pattern: str) -> str:
        """Returns lines that match a pattern in the files of the provided Directory"""
        return await (
            dag.container()
            .from_("alpine:latest")
            .with_mounted_directory("/mnt", directory_arg)
            .with_workdir("/mnt")
            .with_exec(["grep", "-R", pattern, "."])
            .stdout()
        )
